---
title: Broken Pipeline Fixer
emoji: 🔧
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

<div align="center">
  <h1>🔧 Broken Data Pipeline Fixer</h1>
  <p>An OpenEnv-compatible reinforcement learning environment for repairing broken data pipelines.</p>
  
  ![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)
  ![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-8b5cf6)
  ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
</div>

## 🧠 Environment & RL Formulation

This project is formulated as a standard sparse-reward reinforcement learning problem with sequence modifications. The environment acts as an interpreter, applying a pipeline of data operations (e.g., `load_csv`, `clean_nulls`) using a strict schema engine. If any step fails data dependency checks or type validations, the pipeline halts.

### State (Observation Space)
The state object (`obs`) encapsulates:
- `pipeline`: The sequence of current operations.
- `schema_state`: The dynamic dataframe schema evaluated up to the point of failure.
- `error`: The explicit fatal error message (if any), such as "Column 'email' not found."
- `issues_remaining`: Internal metric indicating structural difference against correct solution.

### Action Space
The agent repairs the pipeline using a discrete, structured action space string format:
| Action | Description |
|:---|:---|
| `diagnose` | Request deeper log inspection without mutating state. |
| `reorder` | Automatically sort operations chronologically by operational phase (extract → clean → transform). |
| `swap:<i>:<j>` | Swap step `i` and `j`. |
| `remove:<pos>` | Attempt to delete anomalous constraints or logic. |
| `insert:<pos>:<op>:<k=v>` | Bootstrap missing operations. |
| `fix_param:<pos>:k:v` | Resolve typos or parameter drift. |

### Reward Shaping
- **`+1.00`** — Full completion (pipeline functions optimally & matches target).
- **`+0.30` * `progress`** — Proportional reward for making structural progress.
- **`+0.05`** — Pipeline successfully executes end-to-end without failing.
- **`-0.05`** — Standard penalty per turn to incentivize speed.
- **`-0.10` / `-0.20`** — Inefficient or actively regressive steps.
- **`-0.30`** — Formatting failure (hallucination).

### Scoring
All scores are bounded in the strict `(0.001, 0.999)` envelope to satisfy grader requirements:
- **Correctness (40%)** — Does the pipeline match the correct reference?
- **Efficiency (30%)** — How close to optimal step count?
- **Issues Resolved (15%)** — Fraction of original issues fixed.
- **Error-free Execution (15%)** — Does the pipeline run without errors?

---

## 🚀 Generalizability API

This environment adheres strictly to OpenAI Gym/OpenEnv conventions but incorporates **complete generalization**. You do not need to rely on the built-in tasks!

### Initializing with Custom Pipelines

You can directly construct the environment with your own broken pipeline sequence and target standard:

```python
from env.pipeline_env import DataPipelineEnv

# Fully generalized constructor
env = DataPipelineEnv(
    pipeline=[
        {"op": "load_csv", "params": {"file": "data.csv"}},
        {"op": "filter_rows", "params": {"column": "sale_price", "condition": ">", "value": "100"}}
    ],
    schema={"id": "int", "sales_price": "float"},  # Typo in source column format triggers breakage
    correct_pipeline=[ ... ]
)

# Start sequence exactly as standard
obs, info = env.reset()
```

The system automatically parses structured operations, builds the step validation graph, and calculates error counts based on the custom input!

### OpenEnv Async Client

For integration with the OpenEnv evaluation framework, use `my_env.py`:

```python
from my_env import BrokenPipelineEnv, BrokenPipelineAction

env = BrokenPipelineEnv()
result = await env.reset()
result = await env.step(BrokenPipelineAction(message="diagnose"))
result = await env.step(BrokenPipelineAction(message="fix_param:1:column:purchase_amt"))
await env.close()
```

---

## 🧪 Running the Environment

### Multi-Provider LLM Support

The agent supports multiple LLM providers with automatic detection:

| Provider | Environment Variable | Default Model |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| HuggingFace | `HF_TOKEN` | `Qwen/Qwen2.5-72B-Instruct` |
| Groq | `GROQ_API_KEY` | `llama-3.3-70b-versatile` |
| Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` |

### 1. Terminal Inference & Custom Verification

To benchmark the RL agent locally without touching an API or UI, run `inference.py`.
By default, this will run the built-in `[easy, medium, hard]` tasks:

```bash
python inference.py
```

> [!TIP]
> **Evaluate with your own JSON structure!**
> You can pass an arbitrary setup to verify robustness against unseen data:
> ```bash
> python inference.py --input input_pipeline.json
> ```

Output follows the strict OpenEnv logging format:
```text
[START] task=easy env=broken_pipeline_fixer model=RuleBasedAgent
[STEP] step=1 action=insert:2:rename_column:from=signup_date:to=registered reward=0.25 done=false error=null
[STEP] step=2 action=fix_param:3:column:age reward=1.00 done=true error=null
[END] success=true steps=2 score=0.9990 rewards=0.25,1.00
```
*Note: If no API key is set, `inference.py` gracefully falls back to a deterministic, completely free `RuleBasedAgent` to solve the pipeline deterministically!*

### 2. Interactive Web UI

This project also includes a beautiful, premium Hackathon-ready **Live Demo Environment**.

Start the server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```
Then visit: `http://localhost:7860/`

---

## 🌐 Headless API Evaluation (`POST /run-pipeline`)

For hackathon judging and external evaluation, the system exposes a seamless REST API interface. You can submit entirely arbitrary workflows to the environment without any UI interaction. The RL evaluation agent will attempt to resolve structural issues and return a complete episode trace!

**Endpoint**: `POST /run-pipeline`

**Request Format** (`application/json`):
```json
{
  "pipeline": [
    {"op": "clean_nulls", "params": {"column": "email"}},
    {"op": "load_csv", "params": {}},
    {"op": "save_output", "params": {"format": "csv"}}
  ],
  "schema": ["user_id", "email_address", "revenue"]
}
```

> [!TIP]
> `correct_pipeline` is an optional array parameter. If omitted, the environment will automatically infer a structurally organized baseline pipeline for the agent to optimize towards! Note: The `schema` can be an array of strings or a `{column: type}` dictionary!

**Curl Example**:
```bash
curl -X POST http://localhost:7860/run-pipeline \
     -H "Content-Type: application/json" \
     -d '{
       "pipeline": [
           {"op": "clean_nulls", "params": {"column": "email"}},
           {"op": "load_csv", "params": {}}
       ],
       "schema": ["user_id", "revenue"]
     }'
```

**Response Format**: You will receive an exact trace of the sequence of actions, rewards distributed, issues encountered, and the overall summary. 
```json
{
  "steps": [
    {
      "step_number": 1,
      "action": "reorder",
      "reward": 0.05,
      "issue_detected": "Dataset not loaded.",
      "fix_applied": "Reordered: ['clean_nulls', 'load_csv'] \u2192 ['load_csv', 'clean_nulls']",
      "pipeline_state": [
        {"op": "load_csv", "params": {}},
        {"op": "clean_nulls", "params": {"column": "email"}}
      ]
    }
  ],
  "summary": {
    "total_steps": 1,
    "total_reward": 1.05,
    "issues_fixed": 1,
    "success": true
  }
}
```

---

## 🔌 Extended API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Execute one action |
| `/state` | GET | Current environment state |
| `/metadata` | GET | Environment metadata for discovery |
| `/schema` | GET | Pydantic JSON schemas |
| `/mcp` | POST | MCP protocol stub |
| `/dashboard` | GET | Episode monitoring snapshot |
| `/agent/next_action` | GET | LLM agent decision |
| `/run-pipeline` | POST | Headless evaluation |
| `/upload_custom` | POST | Custom pipeline upload |
| `/parse_to_pipeline` | POST | LLM code-to-pipeline parser |
| `/generate_code` | POST | Pipeline-to-code generator |

---

## ✅ Submission Validation

Run the built-in validator before submitting:

```bash
./validate-submission.sh https://your-space.hf.space .
```

This performs 3 checks:
1. **API Ping** — Tests `/reset` endpoint is live
2. **Docker Build** — Verifies container builds successfully
3. **OpenEnv Validate** — Runs `openenv validate` linting

---

## ⚙️ Project Structure
- `env/` : `DataPipelineEnv` (Gym-like structure)
- `core/` : Internal pipeline runner, evaluator, and strict schema validation graph
- `tasks/` : Difficulty logic, break generator, and scoring
- `server/` : OpenEnv FastAPI bindings and REST endpoints
- `static/` : Interactive front-end application
- `agent.py`: LLM agent logic and RuleBasedAgent implementation logic
- `inference.py`: Multi-provider inference script with OpenEnv logging
- `my_env.py`: Async OpenEnv client wrapper
- `validate-submission.sh`: Submission validator script
- `models.py`: Pydantic data models for all API surfaces

## Setup
```bash
# Docker
docker build -t broken-pipeline-fixer .
docker run -p 7860:7860 broken-pipeline-fixer

# Local
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Inference
export HF_TOKEN=your_token  # or OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY
python inference.py
```
