#!/usr/bin/env python3
"""
Baseline Inference Script — Broken Data Pipeline Fixer

Runs all tasks (easy, medium, hard) with a deterministic action strategy,
producing reproducible scores with the required logging format:
  [START], [STEP], [END]

Supports environment variables:
  API_BASE_URL  — URL of the deployed server (optional; runs locally by default)
  MODEL_NAME    — name of the agent/model (default: "deterministic_baseline")

Usage:
    python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# ── Local imports ─────────────────────────────────────────────────────
from env.pipeline_env import DataPipelineEnv
from tasks.tasks import get_broken_pipeline, get_task, list_tasks, get_correct_pipeline
from tasks.graders import grade_episode

# ── Configuration ─────────────────────────────────────────────────────

MODEL_NAME = os.environ.get("MODEL_NAME", "deterministic_baseline")
API_BASE_URL = os.environ.get("API_BASE_URL", "")

# Deterministic action sequence that can solve all task variants
ACTION_SEQUENCE: List[str] = [
    "remove_invalid",
    "fix_order",
    "add_validate",
    "fix_order",
]


# ──────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────


def log_start(task_id: str, task_info: dict, initial_state: list) -> None:
    """Emit [START] log line."""
    payload = {
        "task_id": task_id,
        "difficulty": task_info["difficulty"],
        "description": task_info["description"],
        "model": MODEL_NAME,
        "initial_pipeline": initial_state,
        "correct_pipeline": get_correct_pipeline(),
        "timestamp": time.time(),
    }
    print(f"[START] {json.dumps(payload)}")


def log_step(
    step_num: int,
    action: str,
    observation: list,
    reward: float,
    done: bool,
    info: dict,
) -> None:
    """Emit [STEP] log line."""
    payload = {
        "step": step_num,
        "action": action,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": {k: _serialise(v) for k, v in info.items()},
    }
    print(f"[STEP] {json.dumps(payload)}")


def log_end(
    task_id: str,
    score: float,
    steps_taken: int,
    final_pipeline: list,
    total_reward: float,
    success: bool,
) -> None:
    """Emit [END] log line."""
    payload = {
        "task_id": task_id,
        "score": score,
        "steps_taken": steps_taken,
        "final_pipeline": final_pipeline,
        "total_reward": round(total_reward, 4),
        "success": success,
        "model": MODEL_NAME,
        "timestamp": time.time(),
    }
    print(f"[END] {json.dumps(payload)}")


def _serialise(obj: Any) -> Any:
    """Make an object JSON-safe."""
    if isinstance(obj, (list, tuple)):
        return [_serialise(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────


def run_episode(task_id: str) -> Dict[str, Any]:
    """Run a single episode for a task using the deterministic strategy.

    Returns a dict with episode results.
    """
    task_info = get_task(task_id)
    broken = get_broken_pipeline(task_id)
    env = DataPipelineEnv(pipeline=broken, max_steps=10)

    state = env.reset()
    log_start(task_id, task_info, state)

    total_reward = 0.0

    for i, action in enumerate(ACTION_SEQUENCE, start=1):
        state, reward, done, info = env.step(action)
        total_reward += reward

        log_step(
            step_num=i,
            action=action,
            observation=state,
            reward=reward,
            done=done,
            info=info,
        )

        if done:
            break

    # Grade the episode
    score = grade_episode(
        task_id=task_id,
        final_pipeline=state,
        steps_taken=env.step_count,
        max_steps=env.max_steps,
    )

    success = state == get_correct_pipeline()
    log_end(
        task_id=task_id,
        score=score,
        steps_taken=env.step_count,
        final_pipeline=state,
        total_reward=total_reward,
        success=success,
    )

    return {
        "task_id": task_id,
        "score": score,
        "steps_taken": env.step_count,
        "final_pipeline": state,
        "total_reward": total_reward,
        "success": success,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    """Run all tasks and print summary."""
    print("=" * 60)
    print("  Broken Data Pipeline Fixer — Baseline Inference")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 60)
    print()

    results: Dict[str, Dict[str, Any]] = {}

    for task_id in list_tasks():
        print(f"--- Task: {task_id.upper()} ---")
        result = run_episode(task_id)
        results[task_id] = result
        print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<10} {'Score':>8} {'Steps':>7} {'Success':>9}")
    print(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*9}")

    total_score = 0.0
    for tid, res in results.items():
        total_score += res["score"]
        flag = "✅" if res["success"] else "❌"
        print(
            f"  {tid:<10} {res['score']:>8.4f} {res['steps_taken']:>7} {flag:>9}"
        )

    avg_score = total_score / len(results) if results else 0.0
    print(f"  {'-'*10} {'-'*8} {'-'*7} {'-'*9}")
    print(f"  {'AVG':<10} {avg_score:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
