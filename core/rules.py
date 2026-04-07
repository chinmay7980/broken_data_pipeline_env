"""
Rules Engine for Data Pipeline Validation.

Defines dependency rules between pipeline steps and provides utility
functions to validate pipeline correctness, detect missing steps,
and identify ordering violations.
"""

from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────
# Canonical pipeline definition
# ──────────────────────────────────────────────────────────────────────

CORRECT_PIPELINE: List[str] = ["ingest", "clean", "transform", "validate", "store"]

VALID_STEPS: set = set(CORRECT_PIPELINE)

# ──────────────────────────────────────────────────────────────────────
# Dependency rules: step → prerequisite that must appear *before* it
# ──────────────────────────────────────────────────────────────────────

DEPENDENCY_RULES: Dict[str, str] = {
    "clean": "ingest",
    "transform": "clean",
    "validate": "transform",
    "store": "validate",
}


# ──────────────────────────────────────────────────────────────────────
# Validation utilities
# ──────────────────────────────────────────────────────────────────────


def validate_pipeline(pipeline: List[str]) -> Tuple[bool, List[str]]:
    """Check whether *pipeline* is fully correct.

    Returns:
        A tuple ``(is_valid, issues)`` where *is_valid* is ``True``
        when the pipeline matches the canonical sequence exactly, and
        *issues* is a list of human-readable problem descriptions.
    """
    issues: List[str] = []

    missing = check_missing_steps(pipeline)
    if missing:
        issues.append(f"Missing steps: {missing}")

    order_violations = check_order_violations(pipeline)
    if order_violations:
        issues.append(f"Order violations: {order_violations}")

    invalid = check_invalid_steps(pipeline)
    if invalid:
        issues.append(f"Invalid steps: {invalid}")

    duplicates = check_duplicate_steps(pipeline)
    if duplicates:
        issues.append(f"Duplicate steps: {duplicates}")

    is_valid = pipeline == CORRECT_PIPELINE
    return is_valid, issues


def check_missing_steps(pipeline: List[str]) -> List[str]:
    """Return canonical steps that are absent from *pipeline*."""
    return [step for step in CORRECT_PIPELINE if step not in pipeline]


def check_order_violations(pipeline: List[str]) -> List[Tuple[str, str]]:
    """Return pairs ``(step, prerequisite)`` where the prerequisite does
    not appear before *step* in *pipeline*.

    Only steps actually present in the pipeline are checked.
    """
    violations: List[Tuple[str, str]] = []
    for step, prerequisite in DEPENDENCY_RULES.items():
        if step in pipeline and prerequisite in pipeline:
            if pipeline.index(prerequisite) > pipeline.index(step):
                violations.append((step, prerequisite))
    return violations


def check_invalid_steps(pipeline: List[str]) -> List[str]:
    """Return steps in *pipeline* that are not part of the canonical set."""
    return [step for step in pipeline if step not in VALID_STEPS]


def check_duplicate_steps(pipeline: List[str]) -> List[str]:
    """Return steps that appear more than once in *pipeline*."""
    seen: set = set()
    duplicates: List[str] = []
    for step in pipeline:
        if step in seen and step not in duplicates:
            duplicates.append(step)
        seen.add(step)
    return duplicates


def get_correct_position(step: str) -> int:
    """Return the 0-based index of *step* in the canonical pipeline.

    Raises ``ValueError`` if the step is not a recognised pipeline stage.
    """
    if step not in VALID_STEPS:
        raise ValueError(f"Unknown pipeline step: {step!r}")
    return CORRECT_PIPELINE.index(step)
