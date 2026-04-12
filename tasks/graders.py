"""
Deterministic graders for the dynamic pipeline environment.

Scores 0.0–1.0 based on:
  - Pipeline correctness  (40%)  — does it run and match the correct pipeline?
  - Step efficiency        (30%)  — how close to optimal step count?
  - Issues resolved        (15%)  — fraction of original issues fixed.
  - Error-free execution   (15%)  — does the pipeline run without errors?
"""

from __future__ import annotations

from typing import Any, Dict, List

from core.rules import count_issues, run_pipeline


def grade_episode(
    task_id: str,
    final_pipeline: List[Dict],
    correct_pipeline: List[Dict],
    initial_schema: Dict[str, str],
    steps_taken: int,
    max_steps: int,
    initial_issues: int,
) -> float:
    """Compute a deterministic score in [0.0, 1.0]."""

    correctness = _score_correctness(final_pipeline, correct_pipeline)
    efficiency = _score_efficiency(steps_taken, max_steps, task_id)
    issues_resolved = _score_issues_resolved(
        final_pipeline, correct_pipeline, initial_schema, initial_issues
    )
    execution = _score_execution(final_pipeline, initial_schema)

    raw = (
        0.40 * correctness
        + 0.30 * efficiency
        + 0.15 * issues_resolved
        + 0.15 * execution
    )

    # Strict (0.001, 0.999) envelope — guarantees the score is NEVER
    # exactly 0.0 or 1.0, satisfying grader strict-bounds checks.
    return round(0.001 + 0.998 * min(1.0, max(0.0, raw)), 4)


def _score_correctness(final: List[Dict], correct: List[Dict]) -> float:
    """1.0 if pipeline matches correct, partial credit otherwise."""
    if len(final) != len(correct):
        # Partial credit based on matching ops in order
        matches = 0
        for i, step in enumerate(correct):
            if i < len(final) and final[i].get("op") == step.get("op"):
                if final[i].get("params") == step.get("params"):
                    matches += 1
                else:
                    matches += 0.5  # Right op, wrong params
        return matches / max(len(correct), 1)

    matches = 0
    for a, b in zip(final, correct):
        if a.get("op") == b.get("op") and a.get("params") == b.get("params"):
            matches += 1
        elif a.get("op") == b.get("op"):
            matches += 0.5
    return matches / max(len(correct), 1)


def _score_efficiency(steps_taken: int, max_steps: int, task_id: str) -> float:
    """Reward fewer steps."""
    optimal = {"easy": 2, "medium": 4, "hard": 6}.get(task_id, 4)
    if steps_taken <= 0:
        return 0.0
    if steps_taken <= optimal:
        return 1.0
    remaining = max_steps - optimal
    if remaining <= 0:
        return 1.0
    overshoot = steps_taken - optimal
    return max(1.0 - (overshoot / remaining), 0.0)


def _score_issues_resolved(
    final: List[Dict],
    correct: List[Dict],
    schema: Dict[str, str],
    initial_issues: int,
) -> float:
    """1.0 if all issues resolved."""
    if initial_issues <= 0:
        return 1.0
    current = count_issues(final, schema, correct)
    remaining = current["total"]
    resolved = max(initial_issues - remaining, 0)
    return resolved / initial_issues


def _score_execution(final: List[Dict], schema: Dict[str, str]) -> float:
    """1.0 if the pipeline runs without errors."""
    success, *_ = run_pipeline(final, schema)
    return 1.0 if success else 0.0
