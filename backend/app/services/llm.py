"""
llm.py

Wraps calls to the Groq API. The LLM is an assistant here, not the ML
engine (spec section 16) -- it only ever produces recommendations/analysis
text or structured JSON that the backend validates before using. It never
generates code that gets executed (spec section 29, rule 3).
"""

import json
from groq import Groq

from app.core.config import GROQ_API_KEY

MODEL = "qwen/qwen3.8-27b"

client = Groq(api_key=GROQ_API_KEY)


def analyze_dataset_profile(profile: dict) -> dict:
    """
    Given a dataset profile (from profiler.py), asks the LLM for
    preprocessing recommendations and potential risks, and returns
    parsed JSON -- NOT yet validated against a strict schema, that
    happens one layer up in main.py so this function stays focused
    on just talking to the LLM.
    """
    summary = {
        "rows": profile["basic_info"]["rows"],
        "columns": profile["basic_info"]["columns"],
        "missing_values": profile["basic_info"]["missing_values"],
        "numerical_columns": profile["column_types"]["numerical_columns"],
        "categorical_columns": profile["column_types"]["categorical_columns"],
    }

    prompt = f"""You are analyzing a dataset profile for a machine learning project.

Dataset summary:
{json.dumps(summary, indent=2)}

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{
  "recommended_preprocessing": {{
    "numerical": ["<short recommendation strings>"],
    "categorical": ["<short recommendation strings>"]
  }},
  "potential_risks": ["<short risk description strings>"]
}}"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temperature -- we want consistent, sensible
                          # recommendations, not creative variation
    )

    raw_text = response.choices[0].message.content
    return json.loads(raw_text)  # can raise json.JSONDecodeError -- caught by the caller

VALID_MODEL_NAMES = {"logistic_regression", "random_forest", "xgboost"}


def create_experiment_plan(user_goal: str, profile: dict) -> dict:
    """
    Given a plain-language goal ("predict whether a loan gets approved")
    plus the dataset profile, asks the LLM to propose a structured
    experiment plan. This output is NEVER executed directly -- main.py
    validates every field against what the system actually supports
    before it's allowed to drive anything.
    """
    columns_info = {
        "numerical_columns": profile["column_types"]["numerical_columns"],
        "categorical_columns": profile["column_types"]["categorical_columns"],
    }

    prompt = f"""A user wants to build a machine learning model with this goal:
"{user_goal}"

Available dataset columns:
{json.dumps(columns_info, indent=2)}

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{
  "task": "classification",
  "target": "<a column name from the categorical_columns list above>",
  "models": ["<one or more of: logistic_regression, random_forest, xgboost>"],
  "preprocessing": {{
    "numerical": ["<short strings>"],
    "categorical": ["<short strings>"]
  }},
  "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"]
}}

The "target" MUST be an exact column name from the list above. The "models" list MUST only contain values from: logistic_regression, random_forest, xgboost."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content
    return json.loads(raw_text)

def analyze_experiment_results(experiments: list) -> str:
    """
    Given a list of completed experiments (already fetched from the
    database), asks the LLM to explain why they performed differently.
    Returns plain text, not JSON -- this one is meant to be read as an
    explanation, not parsed and acted on, so there's no structured schema
    to validate here. The critical constraint (spec section 18) is that
    the LLM only ever sees REAL metrics we already computed -- it has no
    ability to invent numbers, since it never generates any itself.
    """
    summary = [
        {
            "model_name": exp["model_name"],
            "metrics": exp["metrics"],
            "training_time": exp["training_time"],
        }
        for exp in experiments
    ]

    prompt = f"""Here are the results of several machine learning experiments on the same dataset:

{json.dumps(summary, indent=2)}

Explain in 2-4 short paragraphs why these models likely performed differently, based ONLY on the metrics shown above. Do not invent any numbers, features, or details not present in this data. Write in plain, clear language suitable for someone learning ML."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content