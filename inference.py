#!/usr/bin/env python3
"""
Inference Script — Broken Data Pipeline Fixer
===================================
Runs a full episode of pipeline repair automatically.

If HF_TOKEN is set, it will use the LLM-based PipelineFixerAgent.
If HF_TOKEN is NOT set, it gracefully falls back to the deterministic RuleBasedAgent
so the environment can be tested locally out-of-the-box.

Supports providing custom broken pipelines via JSON using the `--input` argument!
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Any

from env.pipeline_env import DataPipelineEnv
from tasks.tasks import get_task_data, list_tasks
from tasks.graders import grade_episode
from agent import PipelineFixerAgent, RuleBasedAgent, API_KEY, MODEL_NAME

BENCHMARK = "broken_pipeline_fixer"


def run_episode(env: DataPipelineEnv, correct_pipeline: List[Dict], initial_schema: Dict[str, str], task_id: str) -> float:
    """Core RL loop for a given initialized environment."""
    obs = env.reset()
    
    print(f"\n" + "="*60)
    print(f"[START] task={task_id} env={BENCHMARK}")
    
    if API_KEY:
        print(f"[INFO]  Agent: LLM ({MODEL_NAME})")
        agent = PipelineFixerAgent(correct_pipeline=correct_pipeline)
    else:
        print(f"[INFO]  Agent: RuleBasedAgent (Fallback)")
        agent = RuleBasedAgent(correct_pipeline=correct_pipeline)

    print(f"[INFO]  Initial Issues: {env._initial_issue_count}")
    print("="*60)

    rewards: List[float] = []
    steps = 0
    success = False
    score = 0.0
    done = False
    
    last_info: Any = None
    last_reward = 0.0

    try:
        while not done:
            action = agent.get_action(obs, steps, last_info, last_reward, done)

            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)

            error = "null"
            if info.get("action_type") == "invalid":
                error = info.get("action_result", {}).get("error", "invalid_action").replace(" ", "_").strip()

            print(
                f"[STEP {steps:02d}] action={action:<40} \n"
                f"          reward={reward:+.2f} | issues_left={info['remaining_issues']} | done={'true ' if done else 'false'} "
            )

            last_info = info
            last_reward = reward

        # Grading
        final_pipeline = [{"op": s["op"], "params": s.get("params", {})} for s in obs["pipeline"]]
        score = grade_episode(
            task_id=task_id,
            final_pipeline=final_pipeline,
            correct_pipeline=correct_pipeline,
            initial_schema=initial_schema,
            steps_taken=steps,
            max_steps=obs.get("max_steps", 10),
            initial_issues=env._initial_issue_count,
        )
        success = info.get("pipeline_correct", False)

    except Exception as exc:
        if steps == 0:
            print(
                f"[STEP 01] action=error \n"
                f"          reward=+0.00 | issues_left=Err | done=true \n"
                f"          error: {exc}"
            )
            steps = 1
            rewards = [0.0]
        score = 0.0
        success = False

    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    
    print("-" * 60)
    print(
        f"[END]   success={'true' if success else 'false'} \n"
        f"        steps={steps}/{obs.get('max_steps', 10)}\n"
        f"        score={score:.4f}/1.0000 \n"
        f"        history=[{rewards_str}]"
    )
    print("=" * 60 + "\n")

    return score


def run_default_task(task_id: str) -> float:
    """Run one of the predefined tasks."""
    env = DataPipelineEnv(task_id=task_id)
    task_data = get_task_data(task_id)
    return run_episode(
        env, 
        correct_pipeline=task_data["correct_pipeline"], 
        initial_schema=task_data["initial_schema"], 
        task_id=task_id
    )


def run_custom_task(filepath: str) -> float:
    """Run a custom JSON pipeline file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    required_keys = ["schema", "broken_pipeline", "correct_pipeline"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"Input JSON missing required key: {k}")
            
    # Initialize env explicitly with custom definitions showing API generalizability
    env = DataPipelineEnv(
        pipeline=data["broken_pipeline"],
        schema=data["schema"],
        correct_pipeline=data["correct_pipeline"]
    )
    
    return run_episode(
        env,
        correct_pipeline=data["correct_pipeline"],
        initial_schema=data["schema"],
        task_id=f"custom_({os.path.basename(filepath)})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Fixer Inference Script")
    parser.add_argument("--input", type=str, help="Path to custom JSON pipeline task file.")
    args = parser.parse_args()

    if not API_KEY:
        print("[WARNING] HF_TOKEN or API_KEY not set. Using RuleBasedAgent fallback.\n")

    if args.input:
        score = run_custom_task(args.input)
        print(f"### FINAL RESULTS ###")
        print(f"Task: Custom File   Score: {score:.4f}\n")
    else:
        scores = {}
        for task_id in list_tasks():
            scores[task_id] = run_default_task(task_id)

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"### FINAL RESULTS ###")
        print(f"Average score: {avg:.4f}\n")
        for tid, sc in scores.items():
            print(f"  {tid.capitalize():<10} : {sc:.4f}")


if __name__ == "__main__":
    main()
