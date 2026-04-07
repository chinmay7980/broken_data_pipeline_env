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

import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from typing import Any, Dict

from models import (
    HealthResponse,
    PipelineObservation,
    PipelineState,
    ResetRequest,
    StepRequest,
    CustomPipelineRequest,
    RawCodeRequest,
    RawCodeResponse,
    CodeGenRequest,
)
from server.pipeline_environment import PipelineEnvironment
from agent import PipelineFixerAgent
from server.parsers import parse_code_to_pipeline, generate_code_from_pipeline

_ACTIVE_AGENTS = {}

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

# Static files
_STATIC_DIR = pathlib.Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the interactive UI."""
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Broken Data Pipeline Fixer</h1><p>Visit <a href='/docs'>/docs</a></p>")


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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/upload_custom")
async def upload_custom(request: CustomPipelineRequest) -> Dict[str, str]:
    """Upload a custom pipeline task."""
    import uuid
    from tasks.tasks import CUSTOM_TASKS
    task_id = f"custom_{uuid.uuid4().hex[:8]}"
    CUSTOM_TASKS[task_id] = {
        "task_id": task_id,
        "template": "custom",
        "initial_schema": request.initial_schema,
        "correct_pipeline": request.correct_pipeline,
        "broken_pipeline": request.broken_pipeline,
        "breaks_applied": [],
        "max_steps": request.max_steps,
    }
    return {"task_id": task_id}


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


@app.get("/agent/next_action")
async def agent_next_action() -> Dict[str, str]:
    """Ask the loaded LLM agent for the next action to fix the pipeline."""
    from tasks.tasks import get_task_data
    
    task_id = env.state.task_id
    if not task_id:
        raise HTTPException(status_code=400, detail="No active episode.")

    if task_id not in _ACTIVE_AGENTS:
        task_data = get_task_data(task_id)
        _ACTIVE_AGENTS[task_id] = PipelineFixerAgent(correct_pipeline=task_data["correct_pipeline"])

    agent = _ACTIVE_AGENTS[task_id]
    
    # We reconstruct the observation exactly like reset()/step() returns it
    obs = {
        "pipeline": env.state.current_pipeline,
        "error": env.state.current_pipeline[-1].get("error") if env.state.current_pipeline else None,
        "schema_state": env.state.schema_state,
        "issues_remaining": env.state.issues_remaining,
        "max_steps": env.state.max_steps
    }
    
    # We don't have perfect tracking of last_info across REST API inherently, 
    # but the agent can figure it out from current_pipeline and issues_remaining
    action = agent.get_action(obs, env.state.step_count)
    return {"action": action}


@app.post("/parse_to_pipeline", response_model=RawCodeResponse)
async def parse_to_pipeline(request: RawCodeRequest) -> RawCodeResponse:
    """Uses LLM to convert raw python/sql code to pipeline JSON format."""
    try:
        parsed = parse_code_to_pipeline(request.code)
        return RawCodeResponse(
            initial_schema=parsed.get("initial_schema", {}),
            pipeline=parsed.get("pipeline", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM parsing failed: {str(e)}")


@app.post("/generate_code")
async def generate_code(request: CodeGenRequest) -> Dict[str, str]:
    """Uses LLM to convert fixed pipeline JSON back into executable python/sql code."""
    try:
        code = generate_code_from_pipeline(request.pipeline, request.language)
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM code gen failed: {str(e)}")


# ──────────────────────────────────────────────────────────────────────
# Direct execution
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m server.app``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
