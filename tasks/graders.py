"""
Deterministic graders for the Broken Data Pipeline Fixer.

Each grader produces a score between 0.0 and 1.0 for a completed episode.
Scoring is based on:
  - Pipeline correctness  (40%)
  - Step efficiency        (30%)
  - Order violations fixed (15%)
  - Missing steps resolved (15%)

All graders are deterministic: same inputs → same score.
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.rules import (
    CORRECT_PIPELINE,
    check_missing_steps,
    check_order_violations,
    validate_pipeline,
)


def grade_episode(
    task_id: str,
    final_pipeline: List[str],
    steps_taken: int,
    max_steps: int,
) -> float:
    """Compute a deterministic score in [0.0, 1.0] for a completed episode.

    Parameters
    ----------
    task_id : str
        The task difficulty ('easy', 'medium', 'hard').
    final_pipeline : list[str]
        Pipeline state when the episode ended.
    steps_taken : int
        Total actions the agent executed.
    max_steps : int
        Episode step budget.

    Returns
    -------
    float
        Score between 0.0 (total failure) and 1.0 (perfect).
    """
    correctness = _score_correctness(final_pipeline)
    efficiency = _score_efficiency(steps_taken, max_steps, task_id)
    order = _score_order(final_pipeline)
    completeness = _score_completeness(final_pipeline)

    # Weighted combination
    score = (
        0.40 * correctness
        + 0.30 * efficiency
        + 0.15 * order
        + 0.15 * completeness
    )

    return round(min(max(score, 0.0), 1.0), 4)


# ──────────────────────────────────────────────────────────────────────
# Component scorers
# ──────────────────────────────────────────────────────────────────────


def _score_correctness(pipeline: List[str]) -> float:
    """1.0 if pipeline matches canonical, else fraction of correct positions."""
    if pipeline == CORRECT_PIPELINE:
        return 1.0

    if not pipeline:
        return 0.0

    # Partial credit: what fraction of the correct pipeline is matched?
    max_len = max(len(pipeline), len(CORRECT_PIPELINE))
    matches = sum(
        1
        for i, step in enumerate(CORRECT_PIPELINE)
        if i < len(pipeline) and pipeline[i] == step
    )
    return matches / max_len


def _score_efficiency(steps_taken: int, max_steps: int, task_id: str) -> float:
    """Reward fewer steps. Optimal steps per difficulty level."""
    optimal = _optimal_steps(task_id)
    if steps_taken <= 0:
        return 0.0
    if steps_taken <= optimal:
        return 1.0
    # Linear decay from 1.0 → 0.0 as steps approach max_steps
    remaining_budget = max_steps - optimal
    if remaining_budget <= 0:
        return 1.0
    overshoot = steps_taken - optimal
    return max(1.0 - (overshoot / remaining_budget), 0.0)


def _score_order(pipeline: List[str]) -> float:
    """1.0 if no order violations, decreasing with violations."""
    violations = check_order_violations(pipeline)
    if not violations:
        return 1.0
    # Each violation costs a fraction
    total_deps = len(pipeline) - 1 if len(pipeline) > 1 else 1
    return max(1.0 - (len(violations) / total_deps), 0.0)


def _score_completeness(pipeline: List[str]) -> float:
    """1.0 if no missing steps, decreasing with each missing step."""
    missing = check_missing_steps(pipeline)
    if not missing:
        return 1.0
    return max(1.0 - (len(missing) / len(CORRECT_PIPELINE)), 0.0)


def _optimal_steps(task_id: str) -> int:
    """Return the minimum number of steps to solve each task.

    Easy:   1 step  (add_validate)
    Medium: 2 steps (fix_order + add_validate)
    Hard:   2 steps (fix_order + add_validate)
    """
    return {"easy": 1, "medium": 2, "hard": 2}.get(task_id, 2)


# ──────────────────────────────────────────────────────────────────────
# Convenience
# ──────────────────────────────────────────────────────────────────────


def grade_all_tasks(results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """Grade multiple task results at once.

    Parameters
    ----------
    results : dict
        Mapping of task_id → {final_pipeline, steps_taken, max_steps}.

    Returns
    -------
    dict
        Mapping of task_id → score.
    """
    scores = {}
    for task_id, data in results.items():
        scores[task_id] = grade_episode(
            task_id=task_id,
            final_pipeline=data["final_pipeline"],
            steps_taken=data["steps_taken"],
            max_steps=data["max_steps"],
        )
    return scores
