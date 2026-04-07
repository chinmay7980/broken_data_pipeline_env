#!/usr/bin/env python3
"""
Inference Script — Broken Data Pipeline Fixer
===================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=broken_pipeline_fixer model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

import os
import sys
from typing import List

from openai import OpenAI

from env.pipeline_env import DataPipelineEnv, VALID_ACTIONS
from tasks.tasks import get_broken_pipeline, get_task, list_tasks
from tasks.graders import grade_episode
from core.rules import CORRECT_PIPELINE

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "broken_pipeline_fixer"

# ──────────────────────────────────────────────────────────────────────
# System prompt for the LLM agent
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert data pipeline debugging agent. Your job is to fix broken data pipelines by choosing repair actions.

CORRECT pipeline (in order): ingest → clean → transform → validate → store

Dependency rules:
- clean requires ingest
- transform requires clean
- validate requires transform
- store requires validate

Available actions (respond with EXACTLY one, nothing else):
- fix_order     : Re-order existing steps to satisfy dependency rules
- add_validate  : Insert any missing steps at their correct positions
- remove_invalid: Remove steps that aren't part of the canonical pipeline

Strategy:
1. First remove any invalid/unknown steps (remove_invalid)
2. Then fix the ordering of remaining steps (fix_order)
3. Then add any missing steps (add_validate)
4. If order is still wrong after adding, fix_order again

IMPORTANT: Reply with ONLY the action name. No explanation, no punctuation, no quotes."""


# ──────────────────────────────────────────────────────────────────────
# Action parsing
# ──────────────────────────────────────────────────────────────────────


def parse_action(response: str) -> str:
    """Extract a valid action from the LLM response text."""
    text = response.strip().lower().strip("'\"` \n\t")

    # Direct match
    for action in VALID_ACTIONS:
        if text == action:
            return action

    # Substring match (LLM may add extra words)
    for action in VALID_ACTIONS:
        if action in text:
            return action

    # Return raw (env will reject it with negative reward)
    return text


# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────


def run_task(task_id: str, client: OpenAI) -> float:
    """Run one episode for a task using the LLM agent.

    Returns the graded score (0.0–1.0).
    """
    task_info = get_task(task_id)
    broken = get_broken_pipeline(task_id)
    env = DataPipelineEnv(pipeline=broken, max_steps=10)
    state = env.reset()

    # ── [START] ───────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0
    done = False

    # Build initial conversation context
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    try:
        while not done:
            # Compose the observation message
            user_msg = (
                f"Current pipeline: {state}\n"
                f"Correct pipeline: {list(CORRECT_PIPELINE)}\n"
                f"Step {steps + 1} of {env.max_steps}. "
                f"Which action do you take?"
            )
            messages.append({"role": "user", "content": user_msg})

            # Ask LLM
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=30,
                temperature=0.0,
            )
            raw = completion.choices[0].message.content or ""
            action = parse_action(raw)

            # Keep conversation history
            messages.append({"role": "assistant", "content": action})

            # Execute in environment
            state, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)

            # Determine error field
            error = "null"
            if action not in VALID_ACTIONS:
                error = f"invalid_action:{action}"
            elif not info.get("applied", True) and info.get("detail", "").startswith("Unknown"):
                error = info.get("detail", "unknown_error").replace(" ", "_")

            # ── [STEP] ───────────────────────────────────────────────
            print(
                f"[STEP] step={steps} action={action} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error}"
            )

            # Feed result back to LLM for next turn
            result_msg = (
                f"Pipeline is now: {state}. "
                f"Reward: {reward:.2f}. Done: {done}."
            )
            messages.append({"role": "user", "content": result_msg})

        # ── Grading ──────────────────────────────────────────────────
        score = grade_episode(
            task_id=task_id,
            final_pipeline=state,
            steps_taken=steps,
            max_steps=env.max_steps,
        )
        success = (state == CORRECT_PIPELINE)

    except Exception as exc:
        # Ensure we always emit at least one [STEP] on failure
        if steps == 0:
            print(
                f"[STEP] step=1 action=error "
                f"reward=0.00 done=true "
                f"error={str(exc).replace(' ', '_')[:100]}"
            )
            steps = 1
            rewards = [0.0]
        score = 0.0
        success = False

    # ── [END] (always emitted) ────────────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}"
    )

    return score


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not API_KEY:
        print(
            "ERROR: Set HF_TOKEN or API_KEY environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    scores = {}
    for task_id in list_tasks():
        scores[task_id] = run_task(task_id, client)

    # Summary to stderr (does not interfere with log parsing)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n# Average score: {avg:.4f}", file=sys.stderr)
    for tid, sc in scores.items():
        print(f"#   {tid}: {sc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
