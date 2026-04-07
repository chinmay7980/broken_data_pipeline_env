"""
Schema-based pipeline engine for the Broken Data Pipeline Fixer.

Simulates multi-step data pipelines with real operations,
schema tracking, and meaningful error messages.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────

Schema = Dict[str, str]  # column_name → column_type
StepDict = Dict[str, Any]  # {"op": str, "params": dict}

COLUMN_TYPES = {"int", "str", "float", "datetime", "bool"}
OUTPUT_FORMATS = {"csv", "parquet", "json"}

# ──────────────────────────────────────────────────────────────────────
# Operation Registry
# ──────────────────────────────────────────────────────────────────────

OPERATION_INFO = {
    "load_csv": {
        "description": "Load data from a CSV file",
        "category": "source",
        "order": 0,
    },
    "clean_nulls": {
        "description": "Remove rows with null values in a column",
        "category": "cleaning",
        "order": 1,
    },
    "rename_column": {
        "description": "Rename a column",
        "category": "transform",
        "order": 2,
    },
    "cast_type": {
        "description": "Change a column's data type",
        "category": "transform",
        "order": 2,
    },
    "filter_rows": {
        "description": "Filter rows based on a condition",
        "category": "transform",
        "order": 3,
    },
    "add_derived": {
        "description": "Add a computed / derived column",
        "category": "transform",
        "order": 3,
    },
    "aggregate": {
        "description": "Group by a column and aggregate",
        "category": "analytics",
        "order": 4,
    },
    "sort_by": {
        "description": "Sort rows by a column",
        "category": "transform",
        "order": 5,
    },
    "validate_schema": {
        "description": "Validate that required columns exist",
        "category": "validation",
        "order": 6,
    },
    "save_output": {
        "description": "Save processed data to a file",
        "category": "sink",
        "order": 7,
    },
}

VALID_OPERATIONS = set(OPERATION_INFO.keys())

CATEGORY_ORDER = {
    "source": 0,
    "cleaning": 1,
    "transform": 2,
    "analytics": 3,
    "validation": 4,
    "sink": 5,
}


# ──────────────────────────────────────────────────────────────────────
# Step validation — checks if an operation can run with given schema
# ──────────────────────────────────────────────────────────────────────


def validate_step(
    op_type: str, params: Dict[str, Any], schema: Schema
) -> Tuple[bool, Optional[str]]:
    """Return ``(True, None)`` if the step can execute, else ``(False, reason)``."""

    if op_type not in VALID_OPERATIONS:
        return False, f"unknown operation '{op_type}'"

    if op_type == "load_csv":
        return True, None

    if op_type == "clean_nulls":
        col = params.get("column", "")
        if col not in schema:
            return False, f"column '{col}' not found. Available: {_cols(schema)}"
        return True, None

    if op_type == "rename_column":
        old = params.get("from", "")
        if old not in schema:
            return False, f"column '{old}' not found. Available: {_cols(schema)}"
        new = params.get("to", "")
        if new in schema:
            return False, f"target column '{new}' already exists"
        return True, None

    if op_type == "cast_type":
        col = params.get("column", "")
        if col not in schema:
            return False, f"column '{col}' not found. Available: {_cols(schema)}"
        target = params.get("to", "")
        if target not in COLUMN_TYPES:
            return False, f"invalid type '{target}'. Valid: {sorted(COLUMN_TYPES)}"
        return True, None

    if op_type == "filter_rows":
        col = params.get("column", "")
        if col not in schema:
            return False, f"column '{col}' not found. Available: {_cols(schema)}"
        return True, None

    if op_type == "add_derived":
        source = params.get("source", "")
        if source not in schema:
            return False, f"source column '{source}' not found. Available: {_cols(schema)}"
        return True, None

    if op_type == "aggregate":
        gcol = params.get("group_by", "")
        acol = params.get("agg_column", "")
        if gcol not in schema:
            return False, f"group_by column '{gcol}' not found. Available: {_cols(schema)}"
        if acol not in schema:
            return False, f"agg_column '{acol}' not found. Available: {_cols(schema)}"
        return True, None

    if op_type == "sort_by":
        col = params.get("column", "")
        if col not in schema:
            return False, f"column '{col}' not found. Available: {_cols(schema)}"
        return True, None

    if op_type == "validate_schema":
        required = params.get("required_columns", [])
        missing = [c for c in required if c not in schema]
        if missing:
            return False, f"missing columns {missing}. Available: {_cols(schema)}"
        return True, None

    if op_type == "save_output":
        fmt = params.get("format", "")
        if fmt not in OUTPUT_FORMATS:
            return False, f"invalid format '{fmt}'. Valid: {sorted(OUTPUT_FORMATS)}"
        return True, None

    return True, None


# ──────────────────────────────────────────────────────────────────────
# Schema transforms — how each operation changes the schema
# ──────────────────────────────────────────────────────────────────────


def apply_schema_transform(
    op_type: str, params: Dict[str, Any], schema: Schema
) -> Schema:
    """Return a new schema after applying the operation."""
    schema = dict(schema)

    if op_type == "load_csv":
        pass  # schema set at init

    elif op_type == "rename_column":
        old = params.get("from", "")
        new = params.get("to", "")
        if old in schema:
            schema[new] = schema.pop(old)

    elif op_type == "cast_type":
        col = params.get("column", "")
        target = params.get("to", "str")
        if col in schema:
            schema[col] = target

    elif op_type == "add_derived":
        name = params.get("name", "derived")
        schema[name] = "float"

    elif op_type == "aggregate":
        gcol = params.get("group_by", "")
        acol = params.get("agg_column", "")
        func = params.get("function", "sum")
        schema = {
            gcol: schema.get(gcol, "str"),
            f"{func}_{acol}": "float",
        }

    return schema


# ──────────────────────────────────────────────────────────────────────
# Pipeline runner — simulates execution and reports errors
# ──────────────────────────────────────────────────────────────────────


def run_pipeline(
    steps: List[StepDict], initial_schema: Schema
) -> Tuple[bool, int, str, Schema, List[Dict[str, Any]]]:
    """Simulate pipeline execution.

    Returns
    -------
    success : bool
    fail_index : int
        Index of first failing step (-1 if success).
    error_msg : str
    final_schema : Schema
    step_results : list[dict]
        Per-step status for the observation.
    """
    schema = dict(initial_schema)
    step_results: List[Dict[str, Any]] = []

    for i, step in enumerate(steps):
        op = step.get("op", "")
        params = step.get("params", {})

        ok, err = validate_step(op, params, schema)

        if not ok:
            step_results.append(
                {"step": i, "op": op, "params": params, "status": "error", "error": err}
            )
            # Mark remaining as pending
            for j in range(i + 1, len(steps)):
                s = steps[j]
                step_results.append(
                    {"step": j, "op": s.get("op", ""), "params": s.get("params", {}), "status": "pending"}
                )
            return (
                False,
                i,
                f"Step {i} ({op}) failed: {err}",
                schema,
                step_results,
            )

        schema = apply_schema_transform(op, params, schema)
        step_results.append(
            {"step": i, "op": op, "params": params, "status": "ok"}
        )

    return True, -1, "", schema, step_results


# ──────────────────────────────────────────────────────────────────────
# Dependency ordering helper
# ──────────────────────────────────────────────────────────────────────


def sort_by_category(steps: List[StepDict]) -> List[StepDict]:
    """Sort steps by their category order (source → sink)."""
    def _key(step: StepDict) -> int:
        op = step.get("op", "")
        info = OPERATION_INFO.get(op, {})
        cat = info.get("category", "transform")
        return CATEGORY_ORDER.get(cat, 2)

    return sorted(steps, key=_key)


def count_issues(
    steps: List[StepDict], initial_schema: Schema, correct_steps: List[StepDict]
) -> Dict[str, Any]:
    """Count all issues in the current pipeline vs. the correct one."""

    issues: Dict[str, Any] = {
        "missing_steps": [],
        "extra_steps": [],
        "wrong_order": [],
        "bad_params": [],
        "total": 0,
    }

    # Missing steps (in correct but not in current)
    current_ops = [s.get("op") for s in steps]
    correct_ops = [s.get("op") for s in correct_steps]

    for i, cs in enumerate(correct_steps):
        op = cs.get("op")
        params = cs.get("params", {})
        # Check if this step exists in current
        found = False
        for s in steps:
            if s.get("op") == op and _params_match(s.get("params", {}), params):
                found = True
                break
        if not found:
            issues["missing_steps"].append({"op": op, "params": params, "correct_pos": i})

    # Extra steps (in current but not in correct)
    for i, s in enumerate(steps):
        op = s.get("op")
        if op not in VALID_OPERATIONS:
            issues["extra_steps"].append({"op": op, "position": i})

    # Wrong order — category violations
    for i in range(len(steps) - 1):
        cat_i = _get_category_order(steps[i])
        cat_j = _get_category_order(steps[i + 1])
        if cat_i > cat_j:
            issues["wrong_order"].append({"positions": [i, i + 1]})

    # Bad params — find steps with wrong parameters
    for i, s in enumerate(steps):
        op = s.get("op")
        params = s.get("params", {})
        for cs in correct_steps:
            if cs.get("op") == op and not _params_match(params, cs.get("params", {})):
                issues["bad_params"].append({
                    "step": i,
                    "op": op,
                    "current_params": params,
                    "expected_params": cs.get("params", {}),
                })
                break

    issues["total"] = (
        len(issues["missing_steps"])
        + len(issues["extra_steps"])
        + len(issues["wrong_order"])
        + len(issues["bad_params"])
    )

    return issues


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _cols(schema: Schema) -> str:
    return str(sorted(schema.keys()))


def _params_match(a: Dict, b: Dict) -> bool:
    """Check if two param dicts are equivalent."""
    return a == b


def _get_category_order(step: StepDict) -> int:
    op = step.get("op", "")
    info = OPERATION_INFO.get(op, {})
    cat = info.get("category", "transform")
    return CATEGORY_ORDER.get(cat, 2)
