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

import json
import os
import sys
from typing import List

from openai import OpenAI

from env.pipeline_env import DataPipelineEnv, VALID_ACTION_TYPES
from tasks.tasks import get_task_data, list_tasks
from tasks.graders import grade_episode
from core.rules import OPERATION_INFO

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

The environment tracks the initial data schema and applies operations iteratively. If a step asks for a missing column, it fails.
Your goal is to match the functionality of the "correct" pipeline.

Available actions (respond with EXACTLY one, nothing else):
- diagnose                           : Generate a detailed diagnostic report (no state change)
- swap:<i>:<j>                       : Swap the step at index <i> with the step at index <j>
- insert:<pos>:<op_type>:<key=value> : Insert a new step at <pos> (e.g. insert:2:clean_nulls:column=email) 
- remove:<pos>                       : Remove the step at index <pos>
- fix_param:<step_idx>:<key>:<value> : Fix a step's parameter (e.g. fix_param:1:from:signup_date)
- reorder                            : Automatically reorder existing steps by dependencies (source -> clean -> transform -> sink)

General Strategy:
1. Initially look at the error message. Does it say a column is missing? 
   Maybe the step is applied before the column is renamed, or maybe a step is entirely missing.
2. If steps are obviously in the wrong order (e.g., sink before transform), use `reorder`.
3. If a step is missing, use `insert`.
4. If a step has a typo in the column name, use `fix_param`.
5. If invalid steps exist, use `remove`.

IMPORTANT: Reply with ONLY the action string in the exact format shown above. No explanation, no punctuation, no markdown quotes."""

# ──────────────────────────────────────────────────────────────────────
# Action parsing
# ──────────────────────────────────────────────────────────────────────

def parse_action(response: str) -> str:
    """Extract a valid action from the LLM response text."""
    text = response.strip().strip("'\"` \n\t")
    
    # Just take the first line to be safe if the model blabbers
    text = text.split("\n")[0].strip()
    return text

# ──────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> float:
    """Run one episode for a task using the LLM agent."""
    env = DataPipelineEnv()
    obs = env.reset(task_id=task_id)

    # Need this to get correct pipeline for grading later
    task_data = get_task_data(task_id)

    # ── [START] ───────────────────────────────────────────────────────
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0
    done = False

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    try:
        while not done:
            # Prepare observation summary
            pipe_str = json.dumps([{"op": s["op"], "params": s["params"]} for s in obs["pipeline"]])
            target_str = json.dumps(task_data["correct_pipeline"])
            schema_str = json.dumps(obs.get("schema", {}))
            err_str = obs.get("error", "None (Pipeline runs successfully)")

            user_msg = (
                f"Current pipeline: {pipe_str}\n"
                f"Correct pipeline: {target_str}\n"
                f"Schema State: {schema_str}\n"
                f"Current Error: {err_str}\n"
                f"Issues Remaining: {obs.get('issues_remaining', 'Unknown')}\n"
                f"Step {steps + 1} of {obs.get('max_steps', 10)}. "
                f"Which action do you take?"
            )
            messages.append({"role": "user", "content": user_msg})

            # Ask LLM
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=50,
                temperature=0.0,
            )
            raw = completion.choices[0].message.content or ""
            action = parse_action(raw)

            # Keep conversation history
            messages.append({"role": "assistant", "content": action})

            # Execute in environment
            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)

            # Determine error field
            error = "null"
            if info.get("action_type") == "invalid":
                error = info.get("action_result", {}).get("error", "invalid_action").replace(" ", "_")

            # ── [STEP] ───────────────────────────────────────────────
            print(
                f"[STEP] step={steps} action={action} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error[:100]}"
            )

            # Feed result back to LLM for next turn
            result_msg = (
                f"Action Result: {info.get('action_result', {}).get('detail', 'N/A')}. "
                f"Reward: {reward:.2f}. Done: {done}."
            )
            messages.append({"role": "user", "content": result_msg})

        # ── Grading ──────────────────────────────────────────────────
        final_pipeline = [{"op": s["op"], "params": s["params"]} for s in obs["pipeline"]]
        score = grade_episode(
            task_id=task_id,
            final_pipeline=final_pipeline,
            correct_pipeline=task_data["correct_pipeline"],
            initial_schema=task_data["initial_schema"],
            steps_taken=steps,
            max_steps=obs.get("max_steps", 10),
            initial_issues=env._initial_issue_count,
        )
        success = info.get("pipeline_correct", False)

    except Exception as exc:
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

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n# Average score: {avg:.4f}", file=sys.stderr)
    for tid, sc in scores.items():
        print(f"#   {tid}: {sc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
