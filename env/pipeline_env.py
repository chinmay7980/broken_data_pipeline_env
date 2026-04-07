"""
DataPipelineEnv — A Gym-compatible reinforcement learning environment
for learning to repair broken data pipelines.

The agent observes the current (possibly broken) pipeline state and
selects repair actions.  The environment returns the classic RL tuple:

    (state, reward, done, info)

No ML model is included — this is *only* the environment.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from core.rules import (
    CORRECT_PIPELINE,
    DEPENDENCY_RULES,
    check_missing_steps,
    check_order_violations,
    validate_pipeline,
)


# ──────────────────────────────────────────────────────────────────────
# Action registry
# ──────────────────────────────────────────────────────────────────────

VALID_ACTIONS = {"add_validate", "fix_order", "remove_invalid"}


class DataPipelineEnv:
    """Gym-like environment for repairing broken data pipelines.

    Parameters
    ----------
    pipeline : list[str]
        The initial (broken) pipeline to be repaired.
    max_steps : int, optional
        Maximum number of steps before the episode is forcibly ended.
        Defaults to ``10``.
    """

    def __init__(self, pipeline: List[str], max_steps: int = 10) -> None:
        self.original_pipeline: List[str] = list(pipeline)
        self.current_pipeline: List[str] = []
        self.done: bool = False
        self.max_steps: int = max_steps
        self.step_count: int = 0

        # Bookkeeping for reward shaping
        self._prev_issue_count: int = 0

        # Kickstart the episode
        self.reset()

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def reset(self) -> List[str]:
        """Reset the environment to the original broken pipeline.

        Returns
        -------
        list[str]
            The initial (broken) pipeline state.
        """
        self.current_pipeline = copy.deepcopy(self.original_pipeline)
        self.done = False
        self.step_count = 0
        self._prev_issue_count = self._count_issues()
        return self._get_state()

    def step(self, action: str) -> Tuple[List[str], float, bool, Dict[str, Any]]:
        """Execute one environment step.

        Parameters
        ----------
        action : str
            One of ``"add_validate"``, ``"fix_order"``, or
            ``"remove_invalid"``.

        Returns
        -------
        tuple[list[str], float, bool, dict]
            ``(state, reward, done, info)``
        """
        if self.done:
            raise RuntimeError(
                "Episode has ended. Call reset() before stepping again."
            )

        info: Dict[str, Any] = {"action": action}

        # ── Apply the action ──────────────────────────────────────────
        action_result = self._apply_action(action)
        info.update(action_result)

        # ── Post-action bookkeeping ───────────────────────────────────
        self.step_count += 1

        is_valid, issues = validate_pipeline(self.current_pipeline)
        info["pipeline_valid"] = is_valid
        info["remaining_issues"] = issues
        info["step_count"] = self.step_count

        reward = self._compute_reward(is_valid, action_result)
        self.done = self._is_done(is_valid)

        info["done_reason"] = (
            "pipeline_correct" if is_valid
            else "max_steps_reached" if self.step_count >= self.max_steps
            else "in_progress"
        )

        # Update cached issue count for next reward calculation
        self._prev_issue_count = self._count_issues()

        return self._get_state(), reward, self.done, info

    # ──────────────────────────────────────────────────────────────────
    # Action logic
    # ──────────────────────────────────────────────────────────────────

    def _apply_action(self, action: str) -> Dict[str, Any]:
        """Dispatch *action* and mutate ``self.current_pipeline``.

        Returns a dict of debug/info metadata describing what happened.
        """
        if action not in VALID_ACTIONS:
            return {
                "applied": False,
                "detail": f"Unknown action: {action!r}. "
                          f"Valid actions: {sorted(VALID_ACTIONS)}",
                "detected_issue": None,
                "fix_applied": None,
            }

        if action == "add_validate":
            return self._action_add_validate()
        elif action == "fix_order":
            return self._action_fix_order()
        elif action == "remove_invalid":
            return self._action_remove_invalid()

        # Unreachable, but satisfies static analysis
        return {"applied": False, "detail": "No-op", "detected_issue": None, "fix_applied": None}

    # ── Individual action implementations ─────────────────────────────

    def _action_add_validate(self) -> Dict[str, Any]:
        """Insert missing steps into the pipeline at their correct positions."""
        missing = check_missing_steps(self.current_pipeline)
        if not missing:
            return {
                "applied": False,
                "detail": "No missing steps to add.",
                "detected_issue": "none",
                "fix_applied": None,
            }

        # Insert each missing step at its canonical position (or as close
        # as possible given the current pipeline length).
        for step in missing:
            correct_idx = CORRECT_PIPELINE.index(step)
            insert_idx = min(correct_idx, len(self.current_pipeline))
            self.current_pipeline.insert(insert_idx, step)

        return {
            "applied": True,
            "detail": f"Added missing steps: {missing}",
            "detected_issue": "missing_steps",
            "fix_applied": f"Inserted {missing} into pipeline",
        }

    def _action_fix_order(self) -> Dict[str, Any]:
        """Re-order the pipeline so that all dependency rules are satisfied."""
        violations = check_order_violations(self.current_pipeline)
        if not violations:
            return {
                "applied": False,
                "detail": "No order violations detected.",
                "detected_issue": "none",
                "fix_applied": None,
            }

        # Sort the pipeline according to canonical order
        step_set = set(self.current_pipeline)
        self.current_pipeline = [
            s for s in CORRECT_PIPELINE if s in step_set
        ]

        return {
            "applied": True,
            "detail": f"Fixed {len(violations)} order violation(s): {violations}",
            "detected_issue": "order_violations",
            "fix_applied": "Re-ordered pipeline to satisfy dependency rules",
        }

    def _action_remove_invalid(self) -> Dict[str, Any]:
        """Remove steps that are not part of the canonical pipeline."""
        from core.rules import check_invalid_steps

        invalid = check_invalid_steps(self.current_pipeline)
        if not invalid:
            return {
                "applied": False,
                "detail": "No invalid steps to remove.",
                "detected_issue": "none",
                "fix_applied": None,
            }

        self.current_pipeline = [
            s for s in self.current_pipeline if s not in invalid
        ]

        return {
            "applied": True,
            "detail": f"Removed invalid steps: {invalid}",
            "detected_issue": "invalid_steps",
            "fix_applied": f"Removed {invalid} from pipeline",
        }

    # ──────────────────────────────────────────────────────────────────
    # Reward & termination helpers
    # ──────────────────────────────────────────────────────────────────

    def _compute_reward(
        self, is_valid: bool, action_result: Dict[str, Any]
    ) -> float:
        """Design a shaped reward signal.

        Reward schedule
        ---------------
        * **+1.0** — pipeline is fully correct.
        * **+0.3** — a partial fix that reduces the number of issues.
        * **-0.1** — action was valid but had no effect (redundant step).
        * **-0.3** — unknown / invalid action.
        * **-0.05** — per-step penalty to encourage efficiency.
        """
        # Large reward for solving the pipeline
        if is_valid:
            return 1.0

        step_penalty = -0.05  # encourage fewer steps

        if not action_result.get("applied"):
            # Invalid or no-op action
            if action_result.get("detail", "").startswith("Unknown action"):
                return -0.3 + step_penalty
            return -0.1 + step_penalty

        # Check whether the action actually reduced issues
        current_issue_count = self._count_issues()
        if current_issue_count < self._prev_issue_count:
            return 0.3 + step_penalty  # partial fix
        else:
            return -0.1 + step_penalty  # action fired but didn't help

    def _count_issues(self) -> int:
        """Count the total number of individual issues in the pipeline."""
        _, issues = validate_pipeline(self.current_pipeline)
        return len(issues)

    def _is_done(self, is_valid: bool) -> bool:
        """Episode ends when the pipeline is correct **or** the step
        budget is exhausted."""
        if is_valid:
            return True
        if self.step_count >= self.max_steps:
            return True
        return False

    # ──────────────────────────────────────────────────────────────────
    # State helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_state(self) -> List[str]:
        """Return a copy of the current pipeline (the observable state)."""
        return list(self.current_pipeline)

    def __repr__(self) -> str:
        return (
            f"DataPipelineEnv(pipeline={self.current_pipeline}, "
            f"step={self.step_count}/{self.max_steps}, done={self.done})"
        )
