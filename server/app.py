"""
FastAPI application for the Broken Data Pipeline Fixer environment.

Exposes HTTP endpoints compatible with the OpenEnv specification:
  POST /reset   — start a new episode
  POST /step    — execute an action
  GET  /state   — get current environment state
  GET  /health  — liveness check

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    HealthResponse,
    PipelineObservation,
    PipelineState,
    ResetRequest,
    StepRequest,
)
from server.pipeline_environment import PipelineEnvironment

# ──────────────────────────────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Broken Data Pipeline Fixer",
    description=(
        "An OpenEnv-compatible RL environment for learning to repair "
        "broken data pipelines through sequential decision-making."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single environment instance (per-worker)
env = PipelineEnvironment()

# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness / readiness check."""
    return HealthResponse()


@app.post("/reset", response_model=PipelineObservation)
async def reset(request: ResetRequest = ResetRequest()) -> PipelineObservation:
    """Reset the environment and start a new episode.

    Accepts an optional ``task_id`` (default ``"easy"``).
    Returns the initial observation with the broken pipeline.
    """
    try:
        obs = env.reset(task_id=request.task_id, seed=request.seed)
        return obs
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=PipelineObservation)
async def step(request: StepRequest) -> PipelineObservation:
    """Execute one action in the environment.

    Accepts an ``action`` string and returns the resulting observation.
    """
    try:
        obs = env.step(action=request.action)
        return obs
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=PipelineState)
async def state() -> PipelineState:
    """Return the current internal state of the environment."""
    return env.state


# ──────────────────────────────────────────────────────────────────────
# Direct execution
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m server.app``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
