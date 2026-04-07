"""
Pydantic models for the Broken Data Pipeline Fixer environment.

Defines the wire-format types for Action, Observation, and State
used across the FastAPI server, client, and inference script.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PipelineAction(BaseModel):
    """An action the agent can take to repair the pipeline.

    Attributes
    ----------
    action : str
        One of: ``"add_validate"``, ``"fix_order"``, ``"remove_invalid"``.
    task_id : str | None
        Optional task identifier for multi-task episodes.
    """

    action: str = Field(
        ...,
        description="Repair action: 'add_validate', 'fix_order', or 'remove_invalid'",
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Optional task ID (easy / medium / hard)",
    )


class PipelineObservation(BaseModel):
    """What the agent observes after taking an action.

    Attributes
    ----------
    pipeline : list[str]
        Current state of the pipeline (ordered list of step names).
    reward : float
        Scalar reward for the last action.
    done : bool
        Whether the episode has ended.
    info : dict
        Debug / diagnostic metadata (detected issue, fix applied, etc.).
    """

    pipeline: List[str] = Field(
        ..., description="Current pipeline state"
    )
    reward: float = Field(
        0.0, description="Reward for the last action"
    )
    done: bool = Field(
        False, description="Whether the episode is finished"
    )
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic metadata"
    )


class PipelineState(BaseModel):
    """Full internal state of the environment (for OpenEnv ``state()``).

    Attributes
    ----------
    episode_id : str
        Unique identifier for the current episode.
    task_id : str
        Which task is loaded (easy / medium / hard).
    step_count : int
        Number of steps taken so far.
    max_steps : int
        Step budget for the episode.
    current_pipeline : list[str]
        Current pipeline state.
    original_pipeline : list[str]
        The broken pipeline at episode start.
    done : bool
        Whether the episode is finished.
    """

    episode_id: str = Field("", description="Unique episode identifier")
    task_id: str = Field("easy", description="Task difficulty level")
    step_count: int = Field(0, description="Steps taken so far")
    max_steps: int = Field(10, description="Step budget")
    current_pipeline: List[str] = Field(
        default_factory=list, description="Current pipeline"
    )
    original_pipeline: List[str] = Field(
        default_factory=list, description="Initial broken pipeline"
    )
    done: bool = Field(False, description="Episode finished?")


class ResetRequest(BaseModel):
    """Request body for the ``/reset`` endpoint."""

    task_id: str = Field(
        "easy",
        description="Task to load: 'easy', 'medium', or 'hard'",
    )
    seed: Optional[int] = Field(
        default=None, description="Optional random seed (unused, deterministic env)"
    )


class StepRequest(BaseModel):
    """Request body for the ``/step`` endpoint."""

    action: str = Field(
        ...,
        description="Repair action: 'add_validate', 'fix_order', or 'remove_invalid'",
    )


class HealthResponse(BaseModel):
    """Response for the ``/health`` endpoint."""

    status: str = "ok"
    environment: str = "broken_pipeline_fixer"
    version: str = "1.0.0"
