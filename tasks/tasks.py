"""
Pipeline templates, breakage generator, and task definitions.

Each task uses a real-world pipeline template and applies randomized
breakage based on difficulty level. Seeds ensure reproducibility.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

from core.rules import VALID_OPERATIONS, StepDict, Schema

# ──────────────────────────────────────────────────────────────────────
# Pipeline templates — real-world data workflows
# ──────────────────────────────────────────────────────────────────────

PIPELINE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "user_etl": {
        "name": "User Data ETL Pipeline",
        "domain": "User data processing",
        "description": "Loads user CSV, cleans nulls, renames columns, casts types, validates, and saves.",
        "initial_schema": {
            "id": "int",
            "name": "str",
            "email": "str",
            "age": "str",
            "signup_date": "str",
            "status": "str",
        },
        "correct_pipeline": [
            {"op": "load_csv", "params": {"file": "users.csv"}},
            {"op": "clean_nulls", "params": {"column": "email"}},
            {"op": "rename_column", "params": {"from": "signup_date", "to": "registered"}},
            {"op": "cast_type", "params": {"column": "age", "to": "int"}},
            {"op": "validate_schema", "params": {"required_columns": ["id", "email", "age", "registered"]}},
            {"op": "save_output", "params": {"format": "parquet"}},
        ],
    },
    "sales_analytics": {
        "name": "Sales Analytics Pipeline",
        "domain": "Sales reporting",
        "description": "Loads sales data, casts numeric types, filters invalid rows, computes totals, sorts, and saves.",
        "initial_schema": {
            "order_id": "int",
            "product": "str",
            "quantity": "str",
            "price": "str",
            "customer": "str",
            "date": "str",
        },
        "correct_pipeline": [
            {"op": "load_csv", "params": {"file": "sales.csv"}},
            {"op": "cast_type", "params": {"column": "quantity", "to": "int"}},
            {"op": "cast_type", "params": {"column": "price", "to": "float"}},
            {"op": "filter_rows", "params": {"column": "quantity", "condition": ">", "value": "0"}},
            {"op": "add_derived", "params": {"name": "total", "source": "price", "operation": "multiply"}},
            {"op": "sort_by", "params": {"column": "total", "order": "desc"}},
            {"op": "save_output", "params": {"format": "csv"}},
        ],
    },
    "log_processing": {
        "name": "Server Log Processing Pipeline",
        "domain": "Server log analysis",
        "description": "Loads server logs, cleans null timestamps, casts to datetime, filters errors, sorts chronologically, and saves.",
        "initial_schema": {
            "timestamp": "str",
            "level": "str",
            "message": "str",
            "source": "str",
            "request_id": "str",
        },
        "correct_pipeline": [
            {"op": "load_csv", "params": {"file": "server_logs.csv"}},
            {"op": "clean_nulls", "params": {"column": "timestamp"}},
            {"op": "cast_type", "params": {"column": "timestamp", "to": "datetime"}},
            {"op": "filter_rows", "params": {"column": "level", "condition": "==", "value": "ERROR"}},
            {"op": "sort_by", "params": {"column": "timestamp", "order": "asc"}},
            {"op": "save_output", "params": {"format": "json"}},
        ],
    },
}


# ──────────────────────────────────────────────────────────────────────
# Breakage functions
# ──────────────────────────────────────────────────────────────────────


def _remove_step(pipeline: List[StepDict], rng: random.Random) -> Dict:
    """Remove a random non-source, non-sink step."""
    removable = [i for i in range(1, len(pipeline) - 1)]
    if not removable:
        return {}
    idx = rng.choice(removable)
    removed = pipeline.pop(idx)
    return {"type": "missing_step", "removed": removed, "original_position": idx}


def _swap_steps(pipeline: List[StepDict], rng: random.Random) -> Dict:
    """Swap two adjacent non-source steps."""
    swappable = list(range(1, len(pipeline) - 1))
    if len(swappable) < 2:
        return {}
    i = rng.choice(swappable[:-1])
    pipeline[i], pipeline[i + 1] = pipeline[i + 1], pipeline[i]
    return {"type": "wrong_order", "swapped": [i, i + 1]}


def _corrupt_param(pipeline: List[StepDict], rng: random.Random) -> Dict:
    """Introduce a typo in a column name parameter."""
    corruptible = [
        i for i in range(1, len(pipeline) - 1)
        if pipeline[i].get("op") in ("clean_nulls", "cast_type", "filter_rows", "rename_column", "sort_by")
    ]
    if not corruptible:
        return {}
    idx = rng.choice(corruptible)
    step = pipeline[idx]
    params = dict(step.get("params", {}))

    # Find a column-name param to corrupt
    for key in ("column", "from", "source"):
        if key in params:
            original = params[key]
            # Simple typo: swap two chars or add an extra char
            corrupted = original[:-1] + rng.choice("xyz_") if len(original) > 1 else original + "_"
            params[key] = corrupted
            pipeline[idx] = {"op": step["op"], "params": params}
            return {
                "type": "bad_param",
                "step_index": idx,
                "param_key": key,
                "original_value": original,
                "corrupted_value": corrupted,
            }
    return {}


def _insert_junk(pipeline: List[StepDict], rng: random.Random) -> Dict:
    """Insert a junk / duplicate step."""
    junk_ops = [
        {"op": "cast_type", "params": {"column": "nonexistent_col", "to": "int"}},
        {"op": "filter_rows", "params": {"column": "fake_column", "condition": ">", "value": "0"}},
        {"op": "clean_nulls", "params": {"column": "unknown_field"}},
    ]
    junk = rng.choice(junk_ops)
    pos = rng.randint(1, len(pipeline) - 1)
    pipeline.insert(pos, junk)
    return {"type": "junk_step", "step": junk, "position": pos}


# ──────────────────────────────────────────────────────────────────────
# Breakage combiner
# ──────────────────────────────────────────────────────────────────────

BREAK_FUNCTIONS = [_remove_step, _swap_steps, _corrupt_param, _insert_junk]


def break_pipeline(
    correct_pipeline: List[StepDict],
    difficulty: str,
    seed: int = 42,
) -> Tuple[List[StepDict], List[Dict]]:
    """Apply random breakage to a correct pipeline.

    Parameters
    ----------
    correct_pipeline : list
        The correct pipeline steps.
    difficulty : str
        ``'easy'``, ``'medium'``, or ``'hard'``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    broken_pipeline : list
        The broken pipeline.
    breaks_applied : list[dict]
        Record of what was broken (for grading).
    """
    rng = random.Random(seed)
    pipeline = copy.deepcopy(correct_pipeline)
    breaks_applied = []

    if difficulty == "easy":
        # 1 break: remove a step
        b = _remove_step(pipeline, rng)
        if b:
            breaks_applied.append(b)

    elif difficulty == "medium":
        # 2 breaks: swap + remove
        b = _swap_steps(pipeline, rng)
        if b:
            breaks_applied.append(b)
        b = _remove_step(pipeline, rng)
        if b:
            breaks_applied.append(b)

    elif difficulty == "hard":
        # 3+ breaks: swap + remove + corrupt param + possibly junk
        b = _swap_steps(pipeline, rng)
        if b:
            breaks_applied.append(b)
        b = _remove_step(pipeline, rng)
        if b:
            breaks_applied.append(b)
        b = _corrupt_param(pipeline, rng)
        if b:
            breaks_applied.append(b)
        if rng.random() > 0.3:
            b = _insert_junk(pipeline, rng)
            if b:
                breaks_applied.append(b)

    return pipeline, breaks_applied


# ──────────────────────────────────────────────────────────────────────
# Task definitions
# ──────────────────────────────────────────────────────────────────────

TASK_CONFIGS = {
    "easy": {
        "template": "user_etl",
        "seed": 42,
        "max_steps": 8,
        "difficulty": 1,
        "description": "User ETL pipeline with a missing step.",
    },
    "medium": {
        "template": "sales_analytics",
        "seed": 123,
        "max_steps": 12,
        "difficulty": 2,
        "description": "Sales analytics pipeline with wrong order and a missing step.",
    },
    "hard": {
        "template": "log_processing",
        "seed": 456,
        "max_steps": 15,
        "difficulty": 3,
        "description": "Log processing pipeline with multiple issues: wrong order, missing step, corrupted parameter, and a junk step.",
    },
}


def list_tasks() -> List[str]:
    """Return task IDs in order."""
    return ["easy", "medium", "hard"]


def get_task(task_id: str) -> Dict[str, Any]:
    """Return task metadata."""
    cfg = TASK_CONFIGS[task_id]
    tmpl = PIPELINE_TEMPLATES[cfg["template"]]
    return {
        "task_id": task_id,
        "difficulty": cfg["difficulty"],
        "description": cfg["description"],
        "domain": tmpl["domain"],
        "template_name": tmpl["name"],
        "max_steps": cfg["max_steps"],
    }


def get_task_data(task_id: str) -> Dict[str, Any]:
    """Return the full task data: broken pipeline, correct pipeline, schema, breaks."""
    cfg = TASK_CONFIGS[task_id]
    tmpl = PIPELINE_TEMPLATES[cfg["template"]]
    correct = tmpl["correct_pipeline"]
    schema = tmpl["initial_schema"]

    broken, breaks = break_pipeline(correct, task_id, seed=cfg["seed"])

    return {
        "task_id": task_id,
        "template": cfg["template"],
        "initial_schema": dict(schema),
        "correct_pipeline": copy.deepcopy(correct),
        "broken_pipeline": broken,
        "breaks_applied": breaks,
        "max_steps": cfg["max_steps"],
    }
