"""
HTTP client for the Broken Data Pipeline Fixer environment.

Provides a simple synchronous interface to interact with the
deployed FastAPI environment server via HTTP requests.
"""

from __future__ import annotations

import os
from typing import Optional

import requests

from models import PipelineObservation, PipelineState


class PipelineFixerClient:
    """Synchronous HTTP client for the Pipeline Fixer environment.

    Parameters
    ----------
    base_url : str | None
        Base URL of the environment server.
        Falls back to the ``API_BASE_URL`` environment variable,
        then ``http://localhost:8000``.
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (
            base_url
            or os.environ.get("API_BASE_URL")
            or "http://localhost:8000"
        ).rstrip("/")

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check if the server is alive."""
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> PipelineObservation:
        """Reset the environment for a new episode.

        Parameters
        ----------
        task_id : str
            Task difficulty: ``'easy'``, ``'medium'``, or ``'hard'``.
        seed : int | None
            Optional seed (environment is deterministic).

        Returns
        -------
        PipelineObservation
        """
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        resp = requests.post(f"{self.base_url}/reset", json=payload, timeout=10)
        resp.raise_for_status()
        return PipelineObservation(**resp.json())

    def step(self, action: str) -> PipelineObservation:
        """Execute one action.

        Parameters
        ----------
        action : str
            One of ``'add_validate'``, ``'fix_order'``, ``'remove_invalid'``.

        Returns
        -------
        PipelineObservation
        """
        resp = requests.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=10,
        )
        resp.raise_for_status()
        return PipelineObservation(**resp.json())

    def state(self) -> PipelineState:
        """Get the current environment state.

        Returns
        -------
        PipelineState
        """
        resp = requests.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return PipelineState(**resp.json())

    # ──────────────────────────────────────────────────────────────────
    # Context manager
    # ──────────────────────────────────────────────────────────────────

    def __enter__(self) -> PipelineFixerClient:
        return self

    def __exit__(self, *args) -> None:
        pass

    def __repr__(self) -> str:
        return f"PipelineFixerClient(base_url={self.base_url!r})"
