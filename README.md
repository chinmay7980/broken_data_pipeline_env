# 🔧 Broken Data Pipeline Fixer

An **OpenEnv-compatible reinforcement learning environment** that simulates broken data pipelines and lets an agent learn to repair them through sequential decision-making.

> **Hugging Face Space:** [`Chinmay2005/broken_pipeline_fixer`](https://huggingface.co/spaces/Chinmay2005/broken_pipeline_fixer)

---

## 📖 Problem Statement

Modern data pipelines fail due to:

| Failure Mode | Example |
|---|---|
| **Missing steps** | `validate` stage absent from the flow |
| **Incorrect ordering** | `transform` runs before `clean` |
| **Invalid transformations** | Unrecognised stages injected into the pipeline |
| **Dependency violations** | Steps run without their prerequisites |

Debugging these failures manually is tedious because of multi-stage dependencies. This project reformulates pipeline repair as a **reinforcement learning** problem so that an agent can learn optimal repair strategies automatically.

---

## 🧠 RL Formulation

### State (Observation)

The agent observes the **current pipeline** — an ordered list of step names representing the data flow:

```python
# Example broken pipeline (medium difficulty)
["ingest", "transform", "clean", "store"]  # transform & clean swapped, validate missing
```

### Actions

| Action | Description |
|---|---|
| `add_validate` | Insert any missing steps at their correct canonical positions |
| `fix_order` | Re-order existing steps to satisfy dependency rules |
| `remove_invalid` | Remove steps that aren't part of the canonical pipeline |

### Reward Function

| Condition | Reward |
|---|---|
| Pipeline fully correct | **+1.0** |
| Partial fix (reduces issues) | **+0.3** |
| Redundant action (no effect) | **−0.1** |
| Invalid/unknown action | **−0.3** |
| Per-step efficiency penalty | **−0.05** |

### Episode Termination

An episode ends when:
- ✅ The pipeline matches the canonical correct pipeline, **or**
- ⏱️ The step budget (`max_steps=10`) is exhausted

---

## 🧩 Task Difficulty Levels

| Level | Broken Pipeline | Issues | Optimal Steps |
|---|---|---|---|
| **Easy** | `["ingest", "clean", "transform", "store"]` | Missing `validate` | 1 |
| **Medium** | `["ingest", "transform", "clean", "store"]` | Wrong order + missing `validate` | 2 |
| **Hard** | `["clean", "ingest", "store"]` | Wrong order + missing `transform` & `validate` | 2 |

### Canonical Correct Pipeline

```python
["ingest", "clean", "transform", "validate", "store"]
```

### Dependency Rules

```
clean      → requires ingest
transform  → requires clean
validate   → requires transform
store      → requires validate
```

---

## 📊 Grading

Each task is scored deterministically on a **0.0–1.0** scale:

| Component | Weight | Description |
|---|---|---|
| **Correctness** | 40% | Does the final pipeline match the canonical one? |
| **Efficiency** | 30% | How close to the optimal step count? |
| **Order** | 15% | Are all dependency rules satisfied? |
| **Completeness** | 15% | Are all required steps present? |

---

## 🏗️ Project Structure

```
project_root/
├── openenv.yaml              # OpenEnv manifest
├── Dockerfile                # Docker containerisation
├── .dockerignore
├── requirements.txt
├── README.md
├── inference.py              # Baseline script with [START]/[STEP]/[END] logging
├── models.py                 # Pydantic Action/Observation/State models
├── client.py                 # HTTP client for remote environments
│
├── server/
│   ├── __init__.py
│   ├── app.py                # FastAPI endpoints (/reset, /step, /state, /health)
│   └── pipeline_environment.py  # OpenEnv Environment wrapper
│
├── core/
│   ├── __init__.py
│   └── rules.py              # Dependency rules & validation engine
│
├── env/
│   ├── __init__.py
│   └── pipeline_env.py       # Core Gym-like DataPipelineEnv
│
└── tasks/
    ├── __init__.py
    ├── tasks.py               # Task definitions (easy/medium/hard)
    └── graders.py             # Deterministic scoring (0.0–1.0)
```

---

## ⚙️ Environment Design

### `DataPipelineEnv` (Core)

| Method | Purpose |
|---|---|
| `reset()` | Restore the broken pipeline; return initial state |
| `step(action)` | Apply action → validate → compute reward → return `(state, reward, done, info)` |
| `_apply_action(action)` | Dispatch to action-specific logic |
| `_compute_reward()` | Shaped reward signal |
| `_is_done()` | Check termination condition |

### API Endpoints (Server)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/reset` | POST | Start new episode (accepts `task_id`) |
| `/step` | POST | Execute action (accepts `action`) |
| `/state` | GET | Get current environment state |

---

## 🚀 Setup & Running

### Prerequisites

- Python 3.8+
- Docker (optional, for containerised deployment)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Baseline Inference (Local)

```bash
python inference.py
```

### Run API Server (Local)

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker Build & Run

```bash
docker build -t broken-pipeline-fixer .
docker run -p 8000:8000 broken-pipeline-fixer
```

### Run Inference Against Remote Server

```bash
API_BASE_URL=https://Chinmay2005-broken-pipeline-fixer.hf.space python inference.py
```

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:8000` | URL of the deployed environment |
| `MODEL_NAME` | `deterministic_baseline` | Name of the agent/model for logging |
| `HF_TOKEN` | — | Hugging Face token for Space deployment |

---

## 📜 Logging Format

The inference script produces structured logs:

```
[START] {"task_id": "easy", "difficulty": 1, "model": "deterministic_baseline", ...}
[STEP]  {"step": 1, "action": "remove_invalid", "observation": [...], "reward": -0.15, ...}
[STEP]  {"step": 2, "action": "fix_order", "observation": [...], "reward": -0.15, ...}
[STEP]  {"step": 3, "action": "add_validate", "observation": [...], "reward": 1.0, ...}
[END]   {"task_id": "easy", "score": 0.8167, "steps_taken": 3, "success": true, ...}
```

---

## 📈 Baseline Scores

Deterministic baseline agent results:

| Task | Score | Steps | Result |
|---|---|---|---|
| Easy | ~0.82 | 3 | ✅ |
| Medium | ~0.87 | 3 | ✅ |
| Hard | ~0.87 | 3 | ✅ |
| **Average** | **~0.85** | — | — |

---

## 🔮 Extensibility

This environment is designed to be **drop-in compatible** with standard RL training loops:

```python
from client import PipelineFixerClient

client = PipelineFixerClient(base_url="https://your-space.hf.space")

obs = client.reset(task_id="hard")
print(obs.pipeline)  # broken pipeline

while not obs.done:
    action = agent.select_action(obs.pipeline)
    obs = client.step(action)
    agent.learn(obs)

print(f"Final score: {obs.info}")
```

---

## 📄 License

MIT
