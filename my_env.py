"""
OpenEnv-compatible async client for the Broken Data Pipeline Fixer.

This wrapper provides the standard reset() / step() / close() interface
that the OpenEnv evaluation framework expects. It communicates with the
FastAPI server via HTTP using httpx.

Usage:
    env = BrokenPipelineEnv()
    result = await env.reset()
    result = await env.step(BrokenPipelineAction(message="diagnose"))
    await env.close()
"""

import os
from dotenv import load_dotenv

# Inject .env variables before any imports that might use them
load_dotenv()
if not os.getenv("HF_TOKEN") and not os.getenv("API_KEY"):
    if os.getenv("GROQ_API_KEY"):
        os.environ["API_KEY"] = os.getenv("GROQ_API_KEY")
        if not os.getenv("API_BASE_URL"):
            os.environ["API_BASE_URL"] = "https://api.groq.com/openai/v1"
            os.environ["MODEL_NAME"] = "llama-3.3-70b-versatile"

from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import httpx
import json

class BrokenPipelineAction(BaseModel):
    message: str


class Observation(BaseModel):
    pipeline: List[Dict[str, Any]]
    error: Optional[str] = None
    schema_state: Dict[str, str] = {}
    issues_remaining: int = 0


class Result(BaseModel):
    observation: Observation
    reward: float = 0.0
    done: bool = False


class BrokenPipelineEnv:
    """Async OpenEnv-compatible wrapper for the Pipeline Fixer FastAPI server."""

    def __init__(self):
        self.base_url = os.getenv("ENV_URL", "http://localhost:7860")

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str]) -> "BrokenPipelineEnv":
        """Factory method matching the OpenEnv standard."""
        return cls()

    async def reset(self) -> Result:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{self.base_url}/reset", json={"task_id": "easy"})
                r.raise_for_status()
                data = r.json()
                
                return Result(
                    observation=Observation(
                        pipeline=data.get("pipeline", []),
                        error=data.get("error"),
                        schema_state=data.get("schema_state", {}),
                        issues_remaining=data.get("issues_remaining", 0)
                    ),
                    reward=0.0,
                    done=False,
                )
        except Exception as e:
            return Result(
                observation=Observation(
                    pipeline=[],
                    error=f"Error connecting to backend: {e}",
                    schema_state={},
                    issues_remaining=0
                ),
                reward=0.0,
                done=True,
            )

    async def step(self, action: BrokenPipelineAction) -> Result:
        text = action.message.strip()

        # The LLM should send a valid action string like "diagnose", "swap:1:2", etc.
        # If it sends natural language, we fall back to "diagnose"
        valid_prefixes = ["diagnose", "swap:", "insert:", "remove:", "fix_param:", "reorder"]
        if not any(text.startswith(p) for p in valid_prefixes):
            text = "diagnose"

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{self.base_url}/step", json={"action": text})
                r.raise_for_status()
                data = r.json()

                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))

                return Result(
                    observation=Observation(
                        pipeline=data.get("pipeline", []),
                        error=data.get("error"),
                        schema_state=data.get("schema_state", {}),
                        issues_remaining=data.get("issues_remaining", 0)
                    ),
                    reward=reward,
                    done=done,
                )
        except Exception as e:
            return Result(
                observation=Observation(
                    pipeline=[],
                    error=f"Error: {e}",
                    schema_state={},
                    issues_remaining=0
                ),
                reward=0.0,
                done=True,
            )

    async def close(self) -> None:
        """FastAPI handles its own lifecycle."""
        pass
