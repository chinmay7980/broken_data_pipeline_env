"""
LLM Agent to autonomously fix broken data pipelines.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from env.pipeline_env import DataPipelineEnv
from tasks.tasks import get_task_data

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

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

def parse_action(response: str) -> str:
    """Extract a valid action from the LLM response text."""
    text = response.strip().strip("'\"` \n\t")
    return text.split("\n")[0].strip()

class PipelineFixerAgent:
    """Agent class retaining conversation history for an episode."""
    
    def __init__(self, correct_pipeline: List[Dict]):
        if not API_KEY:
            raise ValueError("API_KEY or HF_TOKEN is missing")
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        self.correct_pipeline = correct_pipeline
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def get_action(self, obs: Dict[str, Any], steps: int, last_info: Optional[Dict] = None, last_reward: float = 0.0, last_done: bool = False) -> str:
        # If we have feedback from a previous step, feed it back
        if last_info is not None:
            result_msg = (
                f"Action Result: {last_info.get('action_result', {}).get('detail', 'N/A')}. "
                f"Reward: {last_reward:.2f}. Done: {last_done}."
            )
            self.messages.append({"role": "user", "content": result_msg})

        pipe_str = json.dumps([{"op": s["op"], "params": s["params"]} for s in obs["pipeline"]])
        target_str = json.dumps(self.correct_pipeline)
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
        self.messages.append({"role": "user", "content": user_msg})

        completion = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=self.messages,
            max_tokens=50,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or ""
        action = parse_action(raw)

        self.messages.append({"role": "assistant", "content": action})
        return action
