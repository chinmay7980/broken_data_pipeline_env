"""
Task Definitions for the Broken Data Pipeline Fixer.

Each task provides a broken pipeline at a specific difficulty level,
along with metadata describing the nature of the breakage.
"""

from typing import Dict, List, Any

from core.rules import CORRECT_PIPELINE


# ──────────────────────────────────────────────────────────────────────
# Task catalogue
# ──────────────────────────────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": "Missing step — 'validate' is absent from the pipeline.",
        "pipeline": ["ingest", "clean", "transform", "store"],
        "issues": ["missing_step"],
        "difficulty": 1,
    },
    "medium": {
        "description": (
            "Wrong order and missing step — 'transform' and 'clean' are "
            "swapped, and 'validate' is missing."
        ),
        "pipeline": ["ingest", "transform", "clean", "store"],
        "issues": ["wrong_order", "missing_step"],
        "difficulty": 2,
    },
    "hard": {
        "description": (
            "Multiple issues — steps are severely out of order and "
            "'transform' and 'validate' are missing."
        ),
        "pipeline": ["clean", "ingest", "store"],
        "issues": ["wrong_order", "missing_steps"],
        "difficulty": 3,
    },
}


def get_task(name: str) -> Dict[str, Any]:
    """Return a task definition by *name*.

    Raises ``KeyError`` if the task does not exist.
    """
    if name not in TASKS:
        available = ", ".join(sorted(TASKS))
        raise KeyError(f"Unknown task {name!r}. Available tasks: {available}")
    return TASKS[name]


def get_broken_pipeline(name: str) -> List[str]:
    """Return a **copy** of the broken pipeline for the given task."""
    return list(get_task(name)["pipeline"])


def list_tasks() -> List[str]:
    """Return the names of all registered tasks."""
    return list(TASKS.keys())


def get_correct_pipeline() -> List[str]:
    """Return a copy of the canonical correct pipeline."""
    return list(CORRECT_PIPELINE)
