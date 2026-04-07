"""
Translators to convert raw Python/SQL code into structured Pipeline environments, and vice versa.
"""

import json
from typing import Dict, Any, List

from openai import OpenAI
from agent import API_KEY, API_BASE_URL, MODEL_NAME

SYSTEM_PROMPT = """\
You are an expert data engineer translating between raw code (SQL/Python Pandas) and structured pipeline JSON.

The allowed operations in the pipeline JSON are:
- `load_csv`: params: `file`
- `clean_nulls`: params: `column`
- `rename_column`: params: `from`, `to`
- `cast_type`: params: `column`, `to` (e.g. int, float, str, datetime)
- `validate_schema`: params: `required_columns` (list of strings)
- `save_output`: params: `format` (e.g. csv, json, parquet)
- `filter_rows`: params: `column`, `condition` (>, <, ==, !=), `value`
- `add_derived`: params: `name`, `source`, `operation` (sum, multiply, etc.)
- `sort_by`: params: `column`, `order` (asc, desc)

When responding, ALWAYS return pure JSON. No markdown backticks, no explanations.
"""

def get_client() -> OpenAI:
    if not API_KEY:
        raise ValueError("Missing HF_TOKEN or API_KEY")
    return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def parse_code_to_pipeline(code: str) -> Dict[str, Any]:
    """Convert raw SQL/Python code into an initial schema and pipeline ops list."""
    prompt = f"""
Analyze the following code and convert it into a structured Data Pipeline JSON object.
It must contain two keys:
"initial_schema": A dictionary mapping column names to types (str, int, float, datetime) based on the implicit source data.
"pipeline": A list of dictionaries with "op" and "params" according to the allowed operations.

Code:
{code}
"""
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    raw = resp.choices[0].message.content.strip().strip("```json").strip("```").strip()
    return json.loads(raw)

def generate_code_from_pipeline(pipeline: List[Dict], language: str = "python") -> str:
    """Convert a fixed pipeline back into executable code."""
    pipe_str = json.dumps(pipeline, indent=2)
    prompt = f"""
We have standard structured pipeline operations:
{pipe_str}

Translate these sequential operations into clean, idiomatic {language.capitalize()} code.
Do not wrap it in markdown. Just return the raw code text. Do not add explanations.
"""
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    code = resp.choices[0].message.content.strip().strip("```python").strip("```sql").strip("```").strip()
    return code
