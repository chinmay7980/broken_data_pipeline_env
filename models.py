"""
Pydantic models for the Broken Data Pipeline Fixer.

These models define the API contracts for observations, actions,
state, and request/response types used by the FastAPI server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Pipeline Step Detail
# ──────────────────────────────────────────────────────────────────────


class PipelineStepDetail(BaseModel):
    """A single step in the pipeline with its execution status."""

    step: int
    op: str
    params: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # "ok", "error", "pending"
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# API Request / Response Models
# ──────────────────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Body for POST /reset."""

    task_id: str = "easy"
    seed: Optional[int] = None


class StepRequest(BaseModel):
    """Body for POST /step."""

    action: str


class CustomPipelineRequest(BaseModel):
    """Body for POST /upload_custom."""

    initial_schema: Dict[str, str]
    correct_pipeline: List[Dict[str, Any]]
    broken_pipeline: List[Dict[str, Any]]
    max_steps: int = 15


class RawCodeRequest(BaseModel):
    """Body for POST /parse_to_pipeline."""
    code: str
    language: str = "python"


class RawCodeResponse(BaseModel):
    """Response containing parsed components."""
    initial_schema: Dict[str, str]
    pipeline: List[Dict[str, Any]]


class CodeGenRequest(BaseModel):
    """Body for POST /generate_code."""
    pipeline: List[Dict[str, Any]]
    language: str = "python"


class PipelineObservation(BaseModel):
    """Observation returned by reset() and step()."""

    pipeline: List[Dict[str, Any]]
    error: Optional[str] = None
    schema_state: Dict[str, str] = Field(default_factory=dict)
    issues_remaining: int = 0
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class PipelineState(BaseModel):
    """Internal state returned by GET /state."""

    episode_id: str = ""
    task_id: str = "easy"
    step_count: int = 0
    max_steps: int = 10
    current_pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    original_pipeline: List[Dict[str, Any]] = Field(default_factory=list)
    schema_state: Dict[str, str] = Field(default_factory=dict)
    issues_remaining: int = 0
    done: bool = False


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
    environment: str = "broken_pipeline_fixer"
    version: str = "2.0.0"

class RunPipelineRequest(BaseModel):
    """Body for POST /run-pipeline headless endpoint."""
    pipeline: List[Dict[str, Any]]
    schema_def: Any = Field(..., alias="schema")
    correct_pipeline: Optional[List[Dict[str, Any]]] = None

class RunPipelineStepInfo(BaseModel):
    """A single step result recorded during the run."""
    step_number: int
    action: str
    reward: float
    issue_detected: Optional[str] = None
    fix_applied: Optional[str] = None
    pipeline_state: List[Dict[str, Any]]

class RunPipelineSummary(BaseModel):
    """Summary metrics of the evaluation run."""
    total_steps: int
    total_reward: float
    issues_fixed: int
    success: bool

class RunPipelineResponse(BaseModel):
    """Response payload for POST /run-pipeline."""
    steps: List[RunPipelineStepInfo]
    summary: RunPipelineSummary
