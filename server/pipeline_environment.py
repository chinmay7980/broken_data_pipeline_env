"""
OpenEnv-compatible Environment wrapper for the Broken Data Pipeline Fixer.

Wraps the core ``DataPipelineEnv`` and provides the standard
``reset()``, ``step(action)``, ``state()`` interface expected by the
OpenEnv framework and the FastAPI server.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from env.pipeline_env import DataPipelineEnv
from tasks.tasks import get_broken_pipeline, get_task, list_tasks
from models import PipelineObservation, PipelineState


class PipelineEnvironment:
    """OpenEnv-compatible environment for pipeline repair.

    Manages the lifecycle of episodes: loading tasks, resetting state,
    executing actions, and reporting state.
    """

    def __init__(self) -> None:
        self._task_id: str = "easy"
        self._episode_id: str = ""
        self._env: Optional[DataPipelineEnv] = None

    # ──────────────────────────────────────────────────────────────────
    # OpenEnv API
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
    ) -> PipelineObservation:
        """Start a new episode with the given task.

        Parameters
        ----------
        task_id : str
            One of ``'easy'``, ``'medium'``, ``'hard'``.
        seed : int | None
            Ignored (environment is deterministic).

        Returns
        -------
        PipelineObservation
            Initial observation with the broken pipeline.
        """
        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())

        broken = get_broken_pipeline(task_id)
        task_info = get_task(task_id)

        self._env = DataPipelineEnv(pipeline=broken, max_steps=10)
        initial_state = self._env.reset()

        return PipelineObservation(
            pipeline=initial_state,
            reward=0.0,
            done=False,
            info={
                "episode_id": self._episode_id,
                "task_id": self._task_id,
                "task_description": task_info["description"],
                "difficulty": task_info["difficulty"],
                "message": "Environment reset. Pipeline is broken — fix it!",
            },
        )

    def step(self, action: str) -> PipelineObservation:
        """Execute one action in the environment.

        Parameters
        ----------
        action : str
            One of ``'add_validate'``, ``'fix_order'``, ``'remove_invalid'``.

        Returns
        -------
        PipelineObservation
            Observation with updated pipeline, reward, done flag, and info.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        state, reward, done, info = self._env.step(action)

        info["episode_id"] = self._episode_id
        info["task_id"] = self._task_id

        return PipelineObservation(
            pipeline=state,
            reward=reward,
            done=done,
            info=info,
        )

    @property
    def state(self) -> PipelineState:
        """Return the full internal state of the environment."""
        if self._env is None:
            return PipelineState(
                episode_id="",
                task_id=self._task_id,
                step_count=0,
                max_steps=10,
                current_pipeline=[],
                original_pipeline=[],
                done=False,
            )

        return PipelineState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._env.step_count,
            max_steps=self._env.max_steps,
            current_pipeline=list(self._env.current_pipeline),
            original_pipeline=list(self._env.original_pipeline),
            done=self._env.done,
        )
