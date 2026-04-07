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

import sys
from typing import List

from env.pipeline_env import DataPipelineEnv
from tasks.tasks import get_task_data, list_tasks
from tasks.graders import grade_episode
from agent import PipelineFixerAgent, API_KEY, MODEL_NAME

BENCHMARK = "broken_pipeline_fixer"


def run_task(task_id: str) -> float:
    """Run one episode for a task using the LLM agent."""
    env = DataPipelineEnv()
    obs = env.reset(task_id=task_id)
    task_data = get_task_data(task_id)

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    agent = PipelineFixerAgent(correct_pipeline=task_data["correct_pipeline"])

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0
    done = False
    
    last_info = None
    last_reward = 0.0

    try:
        while not done:
            action = agent.get_action(obs, steps, last_info, last_reward, done)

            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)

            error = "null"
            if info.get("action_type") == "invalid":
                error = info.get("action_result", {}).get("error", "invalid_action").replace(" ", "_")

            print(
                f"[STEP] step={steps} action={action} "
                f"reward={reward:.2f} done={'true' if done else 'false'} "
                f"error={error[:100]}"
            )

            last_info = info
            last_reward = reward

        # Grading
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

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps} score={score:.2f} rewards={rewards_str}"
    )

    return score


def main() -> None:
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    scores = {}
    for task_id in list_tasks():
        scores[task_id] = run_task(task_id)

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n# Average score: {avg:.4f}", file=sys.stderr)
    for tid, sc in scores.items():
        print(f"#   {tid}: {sc:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
