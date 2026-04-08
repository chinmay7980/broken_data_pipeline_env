"""
Agents to autonomously fix broken data pipelines.

Includes:
1. PipelineFixerAgent: LLM-based agent.
2. RuleBasedAgent: Fallback heuristic agent for local testing without an API key.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

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
    """LLM Agent retaining conversation history for an episode.
    Uses Hugging Face by default to query the model.
    """
    
    def __init__(self, correct_pipeline: List[Dict]):
        self.correct_pipeline = correct_pipeline
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = None  # Lazy init

    def _init_client(self):
        if not self.client:
            if not API_KEY:
                raise ValueError("API_KEY or HF_TOKEN is missing")
            self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    def get_action(self, obs: Dict[str, Any], steps: int, last_info: Optional[Dict] = None, last_reward: float = 0.0, last_done: bool = False) -> str:
        self._init_client()
        
        # Feed back action results
        if last_info is not None:
            act_res = last_info.get("action_result", {})
            detail = act_res.get("detail", act_res.get("error", "N/A"))
            result_msg = f"Action Result: {detail}. Reward: {last_reward:.2f}. Done: {last_done}."
            self.messages.append({"role": "user", "content": result_msg})

        pipe_str = json.dumps([{"op": s["op"], "params": s.get("params", {})} for s in obs["pipeline"]])
        target_str = json.dumps(self.correct_pipeline)
        schema_str = json.dumps(obs.get("schema_state", {}))
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
            max_tokens=60,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content or ""
        action = parse_action(raw)

        self.messages.append({"role": "assistant", "content": action})
        return action


class RuleBasedAgent:
    """Fallback programmatic agent for local verification without an API token.
    Uses greedy heuristics to systematically fix the pipeline.
    """
    def __init__(self, correct_pipeline: List[Dict]):
        self.correct = [{"op": s["op"], "params": s.get("params", {})} for s in correct_pipeline]
        
    def get_action(self, obs: Dict[str, Any], steps: int, last_info: Optional[Dict] = None, last_reward: float = 0.0, last_done: bool = False) -> str:
        current = [{"op": s["op"], "params": s.get("params", {})} for s in obs["pipeline"]]
        
        # 1. Missing steps
        if len(current) < len(self.correct):
            for i, c_step in enumerate(self.correct):
                if i >= len(current) or current[i]["op"] != c_step["op"] and current[i]["params"] != c_step["params"]:
                    # Try inserting what's missing
                    kv = []
                    for k, v in c_step["params"].items():
                        if isinstance(v, list):
                            v = ",".join(v)
                        kv.append(f"{k}={v}")
                    return f"insert:{i}:{c_step['op']}:{':'.join(kv)}"
                    
        # 2. Extra (junk) steps
        if len(current) > len(self.correct):
            for i, step in enumerate(current):
                if i >= len(self.correct):
                    return f"remove:{i}"
                if step["op"] != self.correct[i]["op"]:
                    # Is it junk or just wrong order? Let's assume junk if it doesn't match the required op set
                    ops = [x["op"] for x in self.correct]
                    if step["op"] not in ops:
                        return f"remove:{i}"

        # 3. Parameter errors (typos)
        for i, (c_step, curr_step) in enumerate(zip(self.correct, current)):
            if c_step["op"] == curr_step["op"]:
                for k, v in c_step["params"].items():
                    if curr_step["params"].get(k) != v:
                        val_str = ",".join(v) if isinstance(v, list) else str(v)
                        return f"fix_param:{i}:{k}:{val_str}"
        
        # 4. Out of order
        ops_current = [s["op"] for s in current]
        ops_correct = [s["op"] for s in self.correct]
        if ops_current != ops_correct:
            # Let's just use the robust built-in action
            return "reorder"

        # 5. Pipeline is identical. Just wait for done or hit diagnose.
        return "diagnose"
