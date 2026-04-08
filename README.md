<div align="center">
  <h1>🔧 Broken Data Pipeline Fixer</h1>
  <p>An OpenEnv-compatible reinforcement learning environment for repairing broken data pipelines.</p>
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

---

## 🧪 Running the Environment

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

Example Output (`python inference.py --input input_pipeline.json`):
```text
[START] task=custom_(input_pipeline.json) env=broken_pipeline_fixer
[INFO]  Agent: RuleBasedAgent (Fallback)
============================================================
[STEP 01] action=fix_param:1:column:purchase_amt         
          reward=+1.00 | issues_left=0 | done=true  
------------------------------------------------------------
[END]   success=true 
        steps=1/15
        score=1.0000/1.0000 

### FINAL RESULTS ###
Task: Custom File   Score: 1.0000
```
*Note: If `HF_TOKEN` isn't set, `inference.py` gracefully falls back to a deterministic, completely free `RuleBasedAgent` to solve the pipeline deterministically!*

### 2. Interactive Web UI

This project also includes a beautiful, premium Hackathon-ready **Live Demo Environment**.

Start the server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```
Then visit: `http://localhost:8000/`

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
curl -X POST http://localhost:8000/run-pipeline \
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

## ⚙️ Project Structure
- `env/` : `DataPipelineEnv` (Gym-like structure)
- `core/` : Internal pipeline runner, evaluator, and strict schema validation graph
- `tasks/` : Difficulty logic, break generator, and scoring
- `server/` : OpenEnv FastAPI bindings and REST endpoints
- `static/` : Interactive front-end application
- `agent.py`: LLM agent logic and RuleBasedAgent implementation logic
