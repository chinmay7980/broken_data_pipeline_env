"""
Dynamic data pipeline environment with schema tracking.

Supports expanded actions: diagnose, swap, insert, remove, fix_param, reorder.
Pipelines are dynamically generated with real data operations and
meaningful error messages.
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
# Action types
# ──────────────────────────────────────────────────────────────────────

VALID_ACTION_TYPES = ["diagnose", "swap", "insert", "remove", "fix_param", "reorder"]


# ──────────────────────────────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────────────────────────────


class DataPipelineEnv:
    """Dynamic schema-aware pipeline repair environment.

    The agent must analyse the broken pipeline, interpret error messages,
    and apply corrective actions (swap, insert, remove, fix_param, reorder).
    """

    def __init__(self, task_id: str = "easy") -> None:
        self.task_id = task_id
        self.max_steps: int = 10
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

    # ──────────────────────────────────────────────────────────────────
    # Core API
    # ──────────────────────────────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset the environment for a new episode.

        Returns the initial observation.
        """
        if task_id:
            self.task_id = task_id

        data = get_task_data(self.task_id)

        self.initial_schema = data["initial_schema"]
        self.correct_pipeline = data["correct_pipeline"]
        self.current_pipeline = copy.deepcopy(data["broken_pipeline"])
        self.original_pipeline = copy.deepcopy(data["broken_pipeline"])
        self.breaks_applied = data["breaks_applied"]
        self.max_steps = data["max_steps"]
        self.step_count = 0
        self.done = False

        issues = count_issues(self.current_pipeline, self.initial_schema, self.correct_pipeline)
        self._initial_issue_count = max(issues["total"], 1)

        return self._build_observation()

    def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""

        if self.done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "Episode already done. Call reset()."}

        # Count issues before action
        issues_before = count_issues(
            self.current_pipeline, self.initial_schema, self.correct_pipeline
        )

        # Parse and apply action
        action_type, action_result = self._apply_action(action_str)

        self.step_count += 1

        # Count issues after action
        issues_after = count_issues(
            self.current_pipeline, self.initial_schema, self.correct_pipeline
        )

        # Check if pipeline runs successfully
        success, fail_idx, error_msg, final_schema, step_results = run_pipeline(
            self.current_pipeline, self.initial_schema
        )

        # Check if pipeline matches correct pipeline
        pipeline_correct = self._pipeline_matches_correct()

        # Compute reward
        reward = self._compute_reward(
            action_type=action_type,
            action_result=action_result,
            issues_before=issues_before["total"],
            issues_after=issues_after["total"],
            pipeline_runs=success,
            pipeline_correct=pipeline_correct,
        )

        # Check termination
        if pipeline_correct:
            self.done = True
            done_reason = "pipeline_correct"
        elif self.step_count >= self.max_steps:
            self.done = True
            done_reason = "max_steps_reached"
        else:
            done_reason = "in_progress"

        obs = self._build_observation()

        info = {
            "action_type": action_type,
            "action_raw": action_str,
            "action_result": action_result,
            "pipeline_runs": success,
            "pipeline_correct": pipeline_correct,
            "error": error_msg if not success else None,
            "issues_before": issues_before["total"],
            "issues_after": issues_after["total"],
            "step_count": self.step_count,
            "done_reason": done_reason,
        }

        return obs, reward, self.done, info

    # ──────────────────────────────────────────────────────────────────
    # Action handling
    # ──────────────────────────────────────────────────────────────────

    def _apply_action(self, action_str: str) -> Tuple[str, Dict[str, Any]]:
        """Parse and apply an action string.

        Returns (action_type, result_dict).
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
                return "invalid", {"error": f"Invalid swap format: '{action_str}'. Use swap:<i>:<j>"}

        elif action_type == "insert" and len(parts) >= 3:
            try:
                pos = int(parts[1])
                op_type = parts[2]
                # Parse remaining parts as key=value params
                params = {}
                for p in parts[3:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        params[k] = v
                    else:
                        # Simple param — try to infer key
                        params["column"] = p
                return "insert", self._action_insert(pos, op_type, params)
            except (ValueError, IndexError):
                return "invalid", {"error": f"Invalid insert format: '{action_str}'"}

        elif action_type == "remove" and len(parts) >= 2:
            try:
                pos = int(parts[1])
                return "remove", self._action_remove(pos)
            except (ValueError, IndexError):
                return "invalid", {"error": f"Invalid remove format: '{action_str}'. Use remove:<pos>"}

        elif action_type == "fix_param" and len(parts) >= 4:
            try:
                step_idx = int(parts[1])
                param_key = parts[2]
                param_val = ":".join(parts[3:])  # Handle values with colons
                return "fix_param", self._action_fix_param(step_idx, param_key, param_val)
            except (ValueError, IndexError):
                return "invalid", {"error": f"Invalid fix_param format: '{action_str}'"}

        elif action_type == "reorder":
            return "reorder", self._action_reorder()

        else:
            return "invalid", {"error": f"Unknown action: '{action_str}'. Valid: {VALID_ACTION_TYPES}"}

    def _action_diagnose(self) -> Dict[str, Any]:
        """Return detailed diagnostic info without changing state."""
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
            "step_results": step_results,
            "issues": issues,
            "current_schema_at_error": schema,
        }

    def _action_swap(self, i: int, j: int) -> Dict[str, Any]:
        """Swap two steps."""
        n = len(self.current_pipeline)
        if not (0 <= i < n and 0 <= j < n):
            return {"applied": False, "detail": f"Invalid indices {i},{j}. Pipeline has {n} steps."}
        if i == j:
            return {"applied": False, "detail": "Cannot swap a step with itself."}

        self.current_pipeline[i], self.current_pipeline[j] = (
            self.current_pipeline[j],
            self.current_pipeline[i],
        )
        return {
            "applied": True,
            "detail": f"Swapped step {i} ({self.current_pipeline[j]['op']}) with step {j} ({self.current_pipeline[i]['op']})",
        }

    def _action_insert(self, pos: int, op_type: str, params: Dict) -> Dict[str, Any]:
        """Insert a new step."""
        if op_type not in VALID_OPERATIONS:
            return {"applied": False, "detail": f"Unknown operation '{op_type}'. Valid: {sorted(VALID_OPERATIONS)}"}

        n = len(self.current_pipeline)
        pos = max(0, min(pos, n))

        new_step = {"op": op_type, "params": params}
        self.current_pipeline.insert(pos, new_step)
        return {
            "applied": True,
            "detail": f"Inserted {op_type} at position {pos} with params {params}",
        }

    def _action_remove(self, pos: int) -> Dict[str, Any]:
        """Remove a step."""
        n = len(self.current_pipeline)
        if not (0 <= pos < n):
            return {"applied": False, "detail": f"Invalid position {pos}. Pipeline has {n} steps."}

        removed = self.current_pipeline.pop(pos)
        return {
            "applied": True,
            "detail": f"Removed step {pos} ({removed['op']})",
            "removed_step": removed,
        }

    def _action_fix_param(self, step_idx: int, param_key: str, param_val: str) -> Dict[str, Any]:
        """Fix a parameter of a step."""
        n = len(self.current_pipeline)
        if not (0 <= step_idx < n):
            return {"applied": False, "detail": f"Invalid step index {step_idx}. Pipeline has {n} steps."}

        step = self.current_pipeline[step_idx]
        params = dict(step.get("params", {}))
        old_val = params.get(param_key, "<missing>")

        # Handle list params (e.g., required_columns)
        if param_key == "required_columns":
            param_val = [v.strip() for v in param_val.split(",")]

        params[param_key] = param_val
        self.current_pipeline[step_idx] = {"op": step["op"], "params": params}
        return {
            "applied": True,
            "detail": f"Fixed step {step_idx} ({step['op']}): {param_key} '{old_val}' → '{param_val}'",
        }

    def _action_reorder(self) -> Dict[str, Any]:
        """Reorder steps by category (source → cleaning → transform → ... → sink)."""
        before = [s["op"] for s in self.current_pipeline]
        self.current_pipeline = sort_by_category(self.current_pipeline)
        after = [s["op"] for s in self.current_pipeline]

        if before == after:
            return {"applied": False, "detail": "Pipeline is already in category order."}
        return {"applied": True, "detail": "Reordered pipeline by operation category."}

    # ──────────────────────────────────────────────────────────────────
    # Reward computation
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
        """Compute shaped reward for the action."""

        if pipeline_correct:
            return 1.0

        if action_type == "invalid":
            return -0.3

        if action_type == "diagnose":
            return -0.05  # Small penalty — information gathering, no fix

        if not action_result.get("applied", False):
            return -0.1  # Action had no effect

        # Progress-based reward
        if issues_after < issues_before:
            progress = (issues_before - issues_after) / max(self._initial_issue_count, 1)
            reward = 0.3 * progress
        elif issues_after > issues_before:
            reward = -0.2  # Made things worse
        else:
            reward = -0.05  # No progress

        # Bonus for pipeline now running
        if pipeline_runs and issues_after == 0:
            reward += 0.5

        # Step penalty
        reward -= 0.05

        return round(reward, 4)

    # ──────────────────────────────────────────────────────────────────
    # Observation building
    # ──────────────────────────────────────────────────────────────────

    def _build_observation(self) -> Dict[str, Any]:
        """Build the observation dict the agent sees."""
        success, fail_idx, error_msg, schema, step_results = run_pipeline(
            self.current_pipeline, self.initial_schema
        )

        issues = count_issues(
            self.current_pipeline, self.initial_schema, self.correct_pipeline
        )

        return {
            "pipeline": step_results,
            "error": error_msg if not success else None,
            "schema": schema,
            "issues_remaining": issues["total"],
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "done": self.done,
        }

    def _pipeline_matches_correct(self) -> bool:
        """Check if current pipeline functionally matches the correct one."""
        if len(self.current_pipeline) != len(self.correct_pipeline):
            return False
        for a, b in zip(self.current_pipeline, self.correct_pipeline):
            if a.get("op") != b.get("op"):
                return False
            if a.get("params") != b.get("params"):
                return False
        return True
