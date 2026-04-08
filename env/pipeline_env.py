"""
DataPipelineEnv — Gym-style reinforcement learning environment for
repairing broken data pipelines.

Observation
-----------
  pipeline      : list[dict]  — current pipeline steps with status
  schema_state  : dict        — data schema at the point of failure
  error         : str | None  — first failing step's error message
  issues_remaining : int      — count of structural issues vs correct pipeline
  step_count    : int
  max_steps     : int
  done          : bool

Actions (string-encoded)
------------------------
  diagnose
  swap:<i>:<j>
  insert:<pos>:<op_type>:<key=value>
  remove:<pos>
  fix_param:<step_idx>:<key>:<value>
  reorder

Reward shaping
--------------
  +1.00   pipeline fully correct
  +0.30 * progress_fraction   each issue resolved
  +0.05   pipeline now executes without error
  -0.05   no-progress action
  -0.10   action had no effect
  -0.20   action made issues worse
  -0.30   invalid / unknown action
  -0.05   per-step efficiency penalty
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from core.rules import (
    OPERATION_INFO,
    VALID_OPERATIONS,
    apply_schema_transform,
    count_issues,
    run_pipeline,
    sort_by_category,
)
from tasks.tasks import get_task_data

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

VALID_ACTION_TYPES = ["diagnose", "swap", "insert", "remove", "fix_param", "reorder"]


# ──────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────


class DataPipelineEnv:
    """Gym-style environment for data pipeline repair.

    The agent receives a broken pipeline and must issue corrective actions
    (swap, insert, remove, fix_param, reorder) to restore it to a fully
    running, schema-valid state that matches the reference correct pipeline.

    Usage
    -----
    >>> env = DataPipelineEnv()
    >>> obs = env.reset("easy")
    >>> obs, reward, done, info = env.step("diagnose")
    >>> obs, reward, done, info = env.step("swap:1:2")
    """

    # Human-readable spec (Gym convention)
    action_space_description: str = (
        "String actions: diagnose | swap:<i>:<j> | insert:<pos>:<op>:<k=v> "
        "| remove:<pos> | fix_param:<idx>:<key>:<val> | reorder"
    )
    observation_space_description: str = (
        "Dict with keys: pipeline (list[dict]), schema_state (dict), "
        "error (str|None), issues_remaining (int), step_count (int), "
        "max_steps (int), done (bool)"
    )

    def __init__(
        self,
        task_id: str = "easy",
        pipeline: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[Dict[str, str]] = None,
        correct_pipeline: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 15,
    ) -> None:
        self.task_id = task_id
        
        if pipeline is not None and schema is not None and correct_pipeline is not None:
            self._validate_pipeline(pipeline)
            self._validate_pipeline(correct_pipeline)
            self._custom_init = True
            self._custom_schema = copy.deepcopy(schema)
            self._custom_pipeline = copy.deepcopy(pipeline)
            self._custom_correct = copy.deepcopy(correct_pipeline)
            self._custom_max_steps = max_steps
        else:
            self._custom_init = False

        self.max_steps: int = 15
        self.step_count: int = 0
        self.done: bool = False

        # Pipeline state
        self.initial_schema: Dict[str, str] = {}
        self.current_pipeline: List[Dict[str, Any]] = []
        self.original_pipeline: List[Dict[str, Any]] = []
        self.correct_pipeline: List[Dict[str, Any]] = []
        self.breaks_applied: List[Dict] = []

        # Tracking
        self._initial_issue_count: int = 0

    @staticmethod
    def _validate_pipeline(pipeline: Any) -> None:
        """Strict validation of structured pipeline format."""
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline must be a list of steps.")
        for i, step in enumerate(pipeline):
            if not isinstance(step, dict):
                raise ValueError(f"Step {i} must be a dictionary. Got: type {type(step)}")
            if "op" not in step:
                raise ValueError(f"Step {i} is missing required key 'op'. Found: {step}")
            if "params" not in step:
                raise ValueError(f"Step {i} is missing required key 'params'. Found: {step}")

    # ──────────────────────────────────────────────────────────────────
    # Core Gym API
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
        pipeline: Optional[List[Dict[str, Any]]] = None,
        schema: Optional[Dict[str, str]] = None,
        correct_pipeline: Optional[List[Dict[str, Any]]] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Parameters
        ----------
        task_id : str, optional
            One of ``"easy"``, ``"medium"``, ``"hard"``.
        seed : int, optional
            Unused (breakage is deterministic per task).
        pipeline : list, optional
            A broken custom pipeline.
        schema : dict, optional
            A custom initial schema.
        correct_pipeline : list, optional
            A target custom pipeline.
        max_steps: int, optional
        
        Returns
        -------
        obs : dict
            Initial observation with the broken pipeline.
        """
        # 1. Custom provided right now
        if pipeline is not None and schema is not None and correct_pipeline is not None:
            self._validate_pipeline(pipeline)
            self._validate_pipeline(correct_pipeline)
            self.initial_schema = copy.deepcopy(schema)
            self.correct_pipeline = copy.deepcopy(correct_pipeline)
            self.current_pipeline = copy.deepcopy(pipeline)
            self.original_pipeline = copy.deepcopy(pipeline)
            self.breaks_applied = []
            self.max_steps = max_steps if max_steps else 15

        # 2. Custom provided at __init__
        elif self._custom_init and not task_id:
            self.initial_schema = copy.deepcopy(self._custom_schema)
            self.correct_pipeline = copy.deepcopy(self._custom_correct)
            self.current_pipeline = copy.deepcopy(self._custom_pipeline)
            self.original_pipeline = copy.deepcopy(self._custom_pipeline)
            self.breaks_applied = []
            self.max_steps = self._custom_max_steps

        # 3. Default Tasks
        else:
            if task_id:
                self.task_id = task_id
            data = get_task_data(self.task_id)
            self.initial_schema = data["initial_schema"]
            self.correct_pipeline = data["correct_pipeline"]
            self.current_pipeline = copy.deepcopy(data["broken_pipeline"])
            self.original_pipeline = copy.deepcopy(data["broken_pipeline"])
            self.breaks_applied = data.get("breaks_applied", [])
            self.max_steps = data["max_steps"]

        self.step_count = 0
        self.done = False

        issues = count_issues(self.current_pipeline, self.initial_schema, self.correct_pipeline)
        self._initial_issue_count = max(issues["total"], 1)

        return self._build_observation()

    def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action.

        Parameters
        ----------
        action_str : str
            A valid action string (see ``action_space_description``).

        Returns
        -------
        obs : dict
            New observation.
        reward : float
            Shaped reward signal.
        done : bool
            Whether the episode has ended.
        info : dict
            Rich debug/explainability dict.
        """
        if self.done:
            obs = self._build_observation()
            return obs, 0.0, True, {
                "error": "Episode already done. Call reset().",
                "issue_detected": None,
                "fix_applied": None,
                "remaining_issues": 0,
            }

        # Snapshot issue count before action
        issues_before = count_issues(
            self.current_pipeline, self.initial_schema, self.correct_pipeline
        )

        # Parse & apply the action
        action_type, action_result = self._apply_action(action_str)

        self.step_count += 1

        # Re-evaluate after action
        issues_after = count_issues(
            self.current_pipeline, self.initial_schema, self.correct_pipeline
        )
        success, fail_idx, error_msg, final_schema, step_results = run_pipeline(
            self.current_pipeline, self.initial_schema
        )
        pipeline_correct = self._pipeline_matches_correct()

        # Reward
        reward = self._compute_reward(
            action_type=action_type,
            action_result=action_result,
            issues_before=issues_before["total"],
            issues_after=issues_after["total"],
            pipeline_runs=success,
            pipeline_correct=pipeline_correct,
        )

        # Termination
        if pipeline_correct:
            self.done = True
            done_reason = "pipeline_correct"
        elif self.step_count >= self.max_steps:
            self.done = True
            done_reason = "max_steps_reached"
        else:
            done_reason = "in_progress"

        obs = self._build_observation()

        # Rich info dict for explainability
        issue_detected = error_msg if not success else None
        fix_applied = action_result.get("detail") if action_result.get("applied") else None

        info = {
            # Explainability
            "issue_detected": issue_detected,
            "fix_applied": fix_applied,
            "remaining_issues": issues_after["total"],
            # Action details
            "action_type": action_type,
            "action_raw": action_str,
            "action_result": action_result,
            # Episode progress
            "pipeline_runs": success,
            "pipeline_correct": pipeline_correct,
            "error": error_msg if not success else None,
            "issues_before": issues_before["total"],
            "issues_after": issues_after["total"],
            "step_count": self.step_count,
            "done_reason": done_reason,
            # Breakage metadata (for grading)
            "breaks_applied": self.breaks_applied,
            "initial_issue_count": self._initial_issue_count,
        }

        return obs, reward, self.done, info

    # ──────────────────────────────────────────────────────────────────
    # Action handlers
    # ──────────────────────────────────────────────────────────────────

    def _apply_action(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """Parse and dispatch an action string.

        Returns ``(action_type, result_dict)``.
        """
        parts = action_str.strip().split(":")
        action_type = parts[0].lower().strip()

        if action_type == "diagnose":
            return "diagnose", self._action_diagnose()

        elif action_type == "swap" and len(parts) >= 3:
            try:
                i, j = int(parts[1]), int(parts[2])
                return "swap", self._action_swap(i, j)
            except (ValueError, IndexError):
                return "invalid", {"applied": False, "error": f"Bad swap format '{action_str}'. Use swap:<i>:<j>"}

        elif action_type == "insert" and len(parts) >= 3:
            try:
                pos = int(parts[1])
                op_type = parts[2]
                params: Dict[str, Any] = {}
                for p in parts[3:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        params[k] = v
                    else:
                        params["column"] = p
                return "insert", self._action_insert(pos, op_type, params)
            except (ValueError, IndexError):
                return "invalid", {"applied": False, "error": f"Bad insert format '{action_str}'"}

        elif action_type == "remove" and len(parts) >= 2:
            try:
                pos = int(parts[1])
                return "remove", self._action_remove(pos)
            except (ValueError, IndexError):
                return "invalid", {"applied": False, "error": f"Bad remove format '{action_str}'. Use remove:<pos>"}

        elif action_type == "fix_param" and len(parts) >= 4:
            try:
                step_idx = int(parts[1])
                param_key = parts[2]
                param_val = ":".join(parts[3:])  # support values containing colons
                return "fix_param", self._action_fix_param(step_idx, param_key, param_val)
            except (ValueError, IndexError):
                return "invalid", {"applied": False, "error": f"Bad fix_param format '{action_str}'"}

        elif action_type == "reorder":
            return "reorder", self._action_reorder()

        else:
            return "invalid", {
                "applied": False,
                "error": f"Unknown action '{action_str}'. Valid types: {VALID_ACTION_TYPES}",
            }

    def _action_diagnose(self) -> Dict[str, Any]:
        """Return detailed diagnostic information without modifying state."""
        success, fail_idx, error_msg, schema, step_results = run_pipeline(
            self.current_pipeline, self.initial_schema
        )
        issues = count_issues(self.current_pipeline, self.initial_schema, self.correct_pipeline)
        return {
            "applied": True,
            "detail": "Diagnostic report generated",
            "pipeline_runs": success,
            "first_error": error_msg,
            "fail_index": fail_idx,
            "issues": issues,
            "schema_at_failure": schema,
        }

    def _action_swap(self, i: int, j: int) -> Dict[str, Any]:
        """Swap steps at positions ``i`` and ``j``."""
        n = len(self.current_pipeline)
        if not (0 <= i < n and 0 <= j < n):
            return {"applied": False, "detail": f"Invalid indices {i},{j} for pipeline with {n} steps."}
        if i == j:
            return {"applied": False, "detail": "Cannot swap a step with itself."}

        self.current_pipeline[i], self.current_pipeline[j] = (
            self.current_pipeline[j],
            self.current_pipeline[i],
        )
        op_i = self.current_pipeline[i]["op"]
        op_j = self.current_pipeline[j]["op"]
        return {
            "applied": True,
            "detail": f"Swapped step {i} ({op_j}) ↔ step {j} ({op_i})",
        }

    def _action_insert(self, pos: int, op_type: str, params: Dict) -> Dict[str, Any]:
        """Insert a new step at ``pos``."""
        if op_type not in VALID_OPERATIONS:
            return {
                "applied": False,
                "detail": f"Unknown operation '{op_type}'. Valid: {sorted(VALID_OPERATIONS)}",
            }
        n = len(self.current_pipeline)
        pos = max(0, min(pos, n))
        new_step = {"op": op_type, "params": params}
        self.current_pipeline.insert(pos, new_step)
        return {"applied": True, "detail": f"Inserted '{op_type}' at position {pos} with params {params}"}

    def _action_remove(self, pos: int) -> Dict[str, Any]:
        """Remove the step at ``pos``."""
        n = len(self.current_pipeline)
        if not (0 <= pos < n):
            return {"applied": False, "detail": f"Invalid position {pos} for pipeline with {n} steps."}
        removed = self.current_pipeline.pop(pos)
        return {"applied": True, "detail": f"Removed step {pos} ('{removed['op']}')"}

    def _action_fix_param(self, step_idx: int, param_key: str, param_val: str) -> Dict[str, Any]:
        """Fix a parameter value of the step at ``step_idx``."""
        n = len(self.current_pipeline)
        if not (0 <= step_idx < n):
            return {"applied": False, "detail": f"Invalid step index {step_idx} for pipeline with {n} steps."}

        step = self.current_pipeline[step_idx]
        params = dict(step.get("params", {}))
        old_val = params.get(param_key, "<missing>")

        # Handle list-valued params (e.g., required_columns=id,email)
        if param_key == "required_columns":
            param_val_parsed: Any = [v.strip() for v in param_val.split(",")]
        else:
            param_val_parsed = param_val

        params[param_key] = param_val_parsed
        self.current_pipeline[step_idx] = {"op": step["op"], "params": params}
        return {
            "applied": True,
            "detail": f"Fixed step {step_idx} ('{step['op']}'): {param_key} '{old_val}' → '{param_val_parsed}'",
        }

    def _action_reorder(self) -> Dict[str, Any]:
        """Sort steps by operation category (source → cleaning → transform → sink)."""
        before = [s["op"] for s in self.current_pipeline]
        self.current_pipeline = sort_by_category(self.current_pipeline)
        after = [s["op"] for s in self.current_pipeline]
        if before == after:
            return {"applied": False, "detail": "Pipeline is already in category order."}
        return {"applied": True, "detail": f"Reordered: {before} → {after}"}

    # ──────────────────────────────────────────────────────────────────
    # Reward function
    # ──────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        action_type: str,
        action_result: Dict,
        issues_before: int,
        issues_after: int,
        pipeline_runs: bool,
        pipeline_correct: bool,
    ) -> float:
        """Compute shaped reward for the action taken.

        Reward table
        ------------
        +1.00  pipeline fully correct (terminal)
        +0.30 * (fixed/initial)  partial fix progress
        +0.05  pipeline newly runs without errors
        -0.05  no-progress (same issue count)
        -0.10  action had no effect (applied=False)
        -0.20  action made things worse
        -0.30  invalid action
        -0.05  per-step efficiency penalty (always applied)
        """
        if pipeline_correct:
            return 1.0

        if action_type == "invalid":
            return -0.3

        if action_type == "diagnose":
            return -0.05  # Information gathering — no fix applied

        if not action_result.get("applied", False):
            return -0.1  # Action parsed but had no effect

        # Progress-based shaping
        if issues_after < issues_before:
            progress = (issues_before - issues_after) / max(self._initial_issue_count, 1)
            reward = 0.3 * progress
        elif issues_after > issues_before:
            reward = -0.2  # Regression
        else:
            reward = -0.05  # No progress

        # Bonus: pipeline now runs cleanly
        if pipeline_runs and issues_after == 0:
            reward += 0.05

        # Per-step efficiency penalty
        reward -= 0.05

        return round(reward, 4)

    # ──────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Dict[str, Any]:
        """Construct the observation dict returned to the agent."""
        success, fail_idx, error_msg, schema, step_results = run_pipeline(
            self.current_pipeline, self.initial_schema
        )
        issues = count_issues(
            self.current_pipeline, self.initial_schema, self.correct_pipeline
        )
        return {
            "pipeline": step_results,
            "schema_state": schema,          # ← consistent key (matches server layer)
            "error": error_msg if not success else None,
            "issues_remaining": issues["total"],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
        }

    # ──────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────

    def _pipeline_matches_correct(self) -> bool:
        """Return True if current pipeline exactly matches the correct pipeline."""
        if len(self.current_pipeline) != len(self.correct_pipeline):
            return False
        for a, b in zip(self.current_pipeline, self.correct_pipeline):
            if a.get("op") != b.get("op") or a.get("params") != b.get("params"):
                return False
        return True
