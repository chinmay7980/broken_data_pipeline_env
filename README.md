---
title: Broken Data Pipeline Fixer
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
---

# 🔧 Broken Data Pipeline Fixer

An **OpenEnv-compatible reinforcement learning environment** that dynamically generates, breaks, and simulates real-world data pipelines. An AI agent learns to repair them by diagnosing schema errors and issuing surgical edits.

> **Hugging Face Space:** [`Chinmay2005/broken_pipeline_fixer`](https://huggingface.co/spaces/Chinmay2005/broken_pipeline_fixer)

---

## 📖 Problem Statement

Modern data pipelines frequently break due to missing columns, incorrect transformations, or incompatible operations. Current AI systems can fix isolated code errors but fail to reason about multi-step data workflows and deep dependencies. 

This project goes beyond simple code fixing: the agent must understand schema evolution and inter-step dependencies.

| Failure Mode | Example |
|---|---|
| **Missing step** | `clean_nulls` removed → downstream step fails on null values |
| **Wrong order** | `cast_type` runs before `clean_nulls` → type casting error |
| **Bad parameter** | Typo in column name: `"emial"` instead of `"email"` |
| **Junk step** | Invalid, extraneous operation injected |

---

## 🧠 Dynamic Pipeline Generation & Schema Engine

Unlike static environments, pipelines here are **dynamically generated**. 

Each task samples from one of 3 real-world templates (User ETL, Sales Analytics, Log Processing) and applies randomized breakage based on the difficulty level.

### Execution & Schema Engine
The environment tracks the data schema at every step. Each operation transforms the schema (e.g. `rename_column`, `add_derived`). If a step attempts an operation on a column that does not exist in the schema at that point in time, it results in a tangible error that the agent must parse.

---

## 🕹️ RL Formulation

### State (Observation)

The agent observes the **current pipeline**, the **error message** if execution failed, and the **schema state**:

```json
{
  "pipeline": [
    {"step": 0, "op": "load_csv", "status": "ok"},
    {"step": 1, "op": "rename_column", "params": {"from": "signup", "to": "reg"}, "status": "ok"},
    {"step": 2, "op": "cast_type", "params": {"column": "signup"}, "status": "error"}
  ],
  "error": "Step 2: column 'signup' not found. Available: ['id', 'reg']",
  "issues_remaining": 1
}
```

### Actions

The agent can perform surgical operations to fix the pipeline:

| Action format | Example |
|---|---|
| `diagnose` | Generate a detailed diagnostic report |
| `swap:<i>:<j>` | `swap:1:2` (Swap steps at positions 1 and 2) |
| `insert:<pos>:<op>:<params>` | `insert:2:clean_nulls:column=email` |
| `remove:<pos>` | `remove:4` (Delete invalid step 4) |
| `fix_param:<idx>:<key>:<val>`| `fix_param:1:from:signup_date` |
| `reorder` | Automatically reorder by dependency category |

### Reward Function

| Condition | Reward |
|---|---|
| Pipeline fully correct | **+1.0** |
| Partial fix (reduces issues count) | **+0.3 * progress_ratio** |
| No effect | **−0.1** |
| Made things worse / Invalid action | **−0.3** |
| Diagnosed | **−0.05** |
| Step penalty | **−0.05** |

---

## 🧩 Task Difficulty Levels

| Level | Template | Max Steps | Breakage Types |
|---|---|---|---|
| **Easy** | User ETL | 8 | 1 break (Missing step OR wrong order) |
| **Medium** | Sales Analytics | 12 | 2 breaks (Order swap + missing step) |
| **Hard** | Log Processing | 15 | 3+ breaks (Swap + missing + corrupt param + random junk) |

---

## 📊 Grading

Each task is scored deterministically on a **0.0–1.0** scale:

| Component | Weight | Description |
|---|---|---|
| **Correctness** | 40% | Does the final pipeline match the correct operations? |
| **Efficiency** | 30% | How close to the optimal step count? |
| **Issues Resolved** | 15% | Fraction of the original issues fixed |
| **Execution** | 15% | Does the pipeline run without throwing schema errors? |

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

Run a baseline Qwen 72B Instruct instance interacting with the env:
```bash
HF_TOKEN=hf_YOUR_TOKEN python inference.py
```

### Run API Server (Local)

Serves the REST API and interactive UI:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Interactive UI available at `http://localhost:8000/`.

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for inference |
| `HF_TOKEN` | — | Hugging Face API key (also used as `API_KEY`) |

---

## 📜 Logging Format

The inference script emits structured stdout logs compatible with OpenEnv validation wrappers:

```
[START] task=easy env=broken_pipeline_fixer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=insert:2:validate_schema:required_columns=id,email reward=0.95 done=true error=null
[END] success=true steps=1 score=1.00 rewards=0.95
```

---

## 🔮 Extensibility

This environment is designed to be **drop-in compatible** with standard RL training loops (using the HTTP client) or with any LLM loop.
