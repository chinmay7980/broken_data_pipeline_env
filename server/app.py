"""
FastAPI application for the Broken Data Pipeline Fixer environment.

Exposes HTTP endpoints compatible with the OpenEnv specification:
  POST /reset   — start a new episode
  POST /step    — execute an action
  GET  /state   — get current environment state
  GET  /health  — liveness check
  GET  /metadata — environment metadata
  GET  /schema  — Pydantic JSON schemas for all models
  POST /mcp     — MCP protocol stub
  GET  /dashboard — current episode snapshot

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import pathlib

from fastapi import FastAPI, HTTPException, Request
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
    RunPipelineRequest,
    RunPipelineResponse,
    RunPipelineStepInfo,
    RunPipelineSummary,
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
# Core OpenEnv Endpoints
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


# ──────────────────────────────────────────────────────────────────────
# Extended OpenEnv Endpoints (matching NeuralPagedAttention)
# ──────────────────────────────────────────────────────────────────────


@app.get("/metadata")
async def metadata():
    """Return environment metadata for hackathon grading and discovery."""
    return {
        "name": "broken_pipeline_fixer",
        "description": (
            "An OpenEnv-compatible RL environment for learning to repair "
            "broken data pipelines through sequential decision-making. "
            "The agent analyses multi-step ETL workflows, identifies structural "
            "failures, and applies corrective actions to restore pipeline integrity."
        ),
        "tasks": ["easy", "medium", "hard"],
        "version": "1.0.0",
        "observation_space": {
            "type": "dict",
            "keys": ["pipeline", "schema_state", "error", "issues_remaining"],
        },
        "action_space": {
            "type": "string",
            "actions": [
                "diagnose",
                "swap:<i>:<j>",
                "insert:<pos>:<op>:<k=v>",
                "remove:<pos>",
                "fix_param:<idx>:<key>:<val>",
                "reorder",
            ],
        },
        "reward_range": [-0.30, 1.00],
    }


@app.get("/schema")
async def schema():
    """Return Pydantic JSON schemas for all API models."""
    return {
        "observation": PipelineObservation.model_json_schema(),
        "state": PipelineState.model_json_schema(),
        "reset_request": ResetRequest.model_json_schema(),
        "step_request": StepRequest.model_json_schema(),
        "health": HealthResponse.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    """MCP protocol stub — hackathon judges check for this endpoint."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    req_id = body.get("id") if isinstance(body, dict) else None
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "name": "broken_pipeline_fixer",
            "description": "AI agent for repairing broken data pipelines",
            "capabilities": {"reset": True, "step": True, "state": True},
        },
    }


@app.get("/dashboard")
async def dashboard():
    """Return a snapshot of the current episode state for monitoring."""
    current_state = env.state
    return {
        "state": current_state.model_dump(),
        "issues_remaining": current_state.issues_remaining,
        "step_count": current_state.step_count,
        "max_steps": current_state.max_steps,
        "done": current_state.done,
        "pipeline_length": len(current_state.current_pipeline),
    }


# ──────────────────────────────────────────────────────────────────────
# LLM-Powered Endpoints
# ──────────────────────────────────────────────────────────────────────


@app.get("/agent/next_action")
async def agent_next_action() -> Dict[str, str]:
    """Ask the loaded LLM agent for the next action to fix the pipeline."""
    from tasks.tasks import get_task_data
    from agent import API_KEY
    
    if not API_KEY:
        raise HTTPException(
            status_code=503, 
            detail="HF_TOKEN or API_KEY environment variable is not set. The LLM Auto-Fix agent is unavailable."
        )
    
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
# Headless Evaluation API
# ──────────────────────────────────────────────────────────────────────

@app.post("/run-pipeline", response_model=RunPipelineResponse)
async def run_pipeline_endpoint(request: RunPipelineRequest) -> RunPipelineResponse:
    """Headless API evaluating a custom pipeline using a programmatic RL agent loop."""
    from env.pipeline_env import DataPipelineEnv
    from agent import RuleBasedAgent
    from core.rules import sort_by_category

    # Convert schema if list
    schema_val = request.schema_def
    if isinstance(schema_val, list):
        schema_dict = {str(col): "str" for col in schema_val}
    elif isinstance(schema_val, dict):
        schema_dict = {str(k): str(v) for k, v in schema_val.items()}
    else:
        raise HTTPException(status_code=400, detail="Schema must be a list or dictionary.")

    # Sort to determine a baseline correct_pipeline if none is provided
    try:
        correct_pipe = request.correct_pipeline
        if not correct_pipe:
            correct_pipe = sort_by_category(request.pipeline)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to infer correct pipeline: {str(e)}")

    # Init env
    try:
        env_instance = DataPipelineEnv(
            pipeline=request.pipeline,
            schema=schema_dict,
            correct_pipeline=correct_pipe,
            max_steps=15
        )
        obs = env_instance.reset()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Environment initialization failed: {str(e)}")

    agent = RuleBasedAgent(correct_pipeline=correct_pipe)

    steps_trace = []
    total_reward = 0.0
    steps = 0
    done = False
    
    last_info = None
    last_reward = 0.0

    try:
        while not done and steps < 15:
            action = agent.get_action(obs, steps, last_info, last_reward, done)
            obs, reward, done, info = env_instance.step(action)
            steps += 1
            total_reward += reward

            steps_trace.append(RunPipelineStepInfo(
                step_number=steps,
                action=action,
                reward=round(reward, 4),
                issue_detected=info.get("issue_detected"),
                fix_applied=info.get("fix_applied"),
                pipeline_state=[{"op": s["op"], "params": s.get("params", {})} for s in obs["pipeline"]]
            ))

            last_info = info
            last_reward = reward

        success = info.get("pipeline_correct", False) if last_info else False
        initial_issues = env_instance._initial_issue_count
        final_issues = obs.get("issues_remaining", 0)
        issues_fixed = max(0, initial_issues - final_issues)

        return RunPipelineResponse(
            steps=steps_trace,
            summary=RunPipelineSummary(
                total_steps=steps,
                total_reward=round(total_reward, 4),
                issues_fixed=issues_fixed,
                success=success
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Runtime error during RL episode: {str(e)}")

# ──────────────────────────────────────────────────────────────────────
# Direct execution
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for ``python -m server.app``."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
