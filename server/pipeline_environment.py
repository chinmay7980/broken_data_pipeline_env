"""
OpenEnv Environment wrapper — delegates to the DataPipelineEnv core.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from core.rules import count_issues, run_pipeline
from env.pipeline_env import DataPipelineEnv
from models import PipelineObservation, PipelineState


class PipelineEnvironment:
    """Wraps DataPipelineEnv and exposes OpenEnv-compatible interface."""

    def __init__(self) -> None:
        self._env = DataPipelineEnv()
        self._episode_id: str = ""
        self._task_id: str = "easy"

    # ── reset ─────────────────────────────────────────────────────────

    def reset(
        self, task_id: str = "easy", seed: Optional[int] = None
    ) -> PipelineObservation:
        self._episode_id = str(uuid.uuid4())
        self._task_id = task_id

        from tasks.tasks import get_task

        task_meta = get_task(task_id)

        obs = self._env.reset(task_id=task_id)

        return PipelineObservation(
            pipeline=obs["pipeline"],
            error=obs.get("error"),
            schema_state=obs.get("schema_state", {}),
            issues_remaining=obs.get("issues_remaining", 0),
            reward=0.0,
            done=False,
            info={
                "episode_id": self._episode_id,
                "task_id": task_id,
                "task_description": task_meta["description"],
                "domain": task_meta["domain"],
                "template": task_meta["template_name"],
                "difficulty": task_meta["difficulty"],
                "message": "Environment reset. Analyse the broken pipeline and fix it!",
                "available_actions": [
                    "diagnose",
                    "swap:<i>:<j>",
                    "insert:<pos>:<op_type>:<key>=<value>",
                    "remove:<pos>",
                    "fix_param:<step>:<key>:<value>",
                    "reorder",
                ],
            },
        )

    # ── step ──────────────────────────────────────────────────────────

    def step(self, action: str) -> PipelineObservation:
        if self._env.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        obs, reward, done, info = self._env.step(action)

        info["episode_id"] = self._episode_id
        info["task_id"] = self._task_id

        return PipelineObservation(
            pipeline=obs["pipeline"],
            error=obs.get("error"),
            schema_state=obs.get("schema_state", {}),
            issues_remaining=obs.get("issues_remaining", 0),
            reward=round(reward, 4),
            done=done,
            info=info,
        )

    # ── state ─────────────────────────────────────────────────────────

    @property
    def state(self) -> PipelineState:
        success, fail_idx, err, schema, _ = run_pipeline(
            self._env.current_pipeline, self._env.initial_schema
        )
        issues = count_issues(
            self._env.current_pipeline, self._env.initial_schema, self._env.correct_pipeline
        )
        return PipelineState(
            episode_id=self._episode_id,
            task_id=self._task_id,
            step_count=self._env.step_count,
            max_steps=self._env.max_steps,
            current_pipeline=self._env.current_pipeline,
            original_pipeline=self._env.original_pipeline,
            schema_state=schema,
            issues_remaining=issues["total"],
            done=self._env.done,
        )
