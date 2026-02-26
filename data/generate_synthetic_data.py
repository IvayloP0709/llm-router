import json 
import random 
from dataclasses import dataclass, asdict 
from typing import List, Dict
import pandas as pd
from google import genai
from dotenv import load_dotenv
import os 
import time

# fetch api key
load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

@dataclass
class QueryRow:
    query: str 
    complexity: int       # 1-5
    task_type: str        # translation, code, reasoning, etc.
    domain: str           # general, technical, etc.
    has_code: int         # 0/1
    is_math: int          # 0/1 
    optimal_model: str    # gemini, gpt-4, etc.
    rationale: str        # short reason

def build_generation_prompt(batch_size:int, seed:int) -> str:
    prompt = f"""
You are generating synthetic training data. Return ONLY a JSON array with exactly {batch_size} objects.

Each object must contain:
query (string), complexity (1-5), task_type, domain, has_code (0/1), is_math (0/1),
optimal_model, rationale (<= 1 sentence).

Allowed values:
task_type = ["translation","summarization","qa","code","debugging","reasoning","creative","analysis"]
domain = ["general","science","finance","medical","legal","education","tech"]
optimal_model = ["gemini-flash","gpt-4o","o3-mini"]

Balance constraints:
- optimal_model: ~40% gemini-flash, ~40% gpt-4o, ~20% o3-mini
- no task_type or domain above 25% of the batch
- include at least 2 examples of each task_type and domain

Rules:
- no duplicate or near-duplicate queries
- vary length and style; include short and long queries
- has_code=1 only if code is clearly implied
- is_math=1 only if explicit math reasoning is needed

Use seed {seed} to ensure reproducibility and vary outputs.

Return JSON only. Example item:
{{"query":"Translate 'hello' into French","complexity":1,"task_type":"translation","domain":"general","has_code":0,"is_math":0,"optimal_model":"gemini-flash","rationale":"Simple translation is low complexity."}}
"""
    return prompt


def call_llm(prompt: str) -> str:
    """
    Call the language model API to generate synthetic data based on the provided prompt.
    
    Args:
        prompt (str): The prompt to send to the language model.

    Returns:
        str: The raw response from the language model.
    """
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return resp.text

def parse_rows(raw: str) -> list[Dict]:
    """
    Parse the raw response from the language model into a list of dictionaries representing query rows.
    
    Args:
        raw (str): The raw response from the language model.

    Returns:
        list[Dict]: A list of dictionaries, each representing a query row.
    """
    allowed_task_types = {
        "translation", "summarization", "qa", "code",
        "debugging", "reasoning", "creative", "analysis"
    }
    allowed_domains = {
        "general", "science", "finance", "medical",
        "legal", "education", "tech"
    }
    allowed_models = {"gemini-flash", "gpt-4o", "o3-mini"}

    required_keys = {
        "query", "complexity", "task_type", "domain",
        "has_code", "is_math", "optimal_model", "rationale"
    }

    text = (raw or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not find a valid JSON array in model output.")
        parsed = json.loads(text[start:end + 1])

    if isinstance(parsed, dict):
        if "rows" in parsed and isinstance(parsed["rows"], list):
            parsed = parsed["rows"]
        elif "data" in parsed and isinstance(parsed["data"], list):
            parsed = parsed["data"]
        else:
            raise ValueError("Expected a JSON array of row objects.")

    if not isinstance(parsed, list):
        raise ValueError("Expected parsed output to be a list of row objects.")

    cleaned_rows: list[Dict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not required_keys.issubset(item.keys()):
            continue

        query = str(item["query"]).strip()
        rationale = str(item["rationale"]).strip()
        task_type = str(item["task_type"]).strip().lower()
        domain = str(item["domain"]).strip().lower()
        optimal_model = str(item["optimal_model"]).strip().lower()

        try:
            complexity = int(item["complexity"])
            has_code = int(item["has_code"])
            is_math = int(item["is_math"])
        except (TypeError, ValueError):
            continue

        if not query:
            continue
        if complexity < 1 or complexity > 5:
            continue
        if has_code not in (0, 1) or is_math not in (0, 1):
            continue
        if task_type not in allowed_task_types:
            continue
        if domain not in allowed_domains:
            continue
        if optimal_model not in allowed_models:
            continue

        cleaned_rows.append({
            "query": query,
            "complexity": complexity,
            "task_type": task_type,
            "domain": domain,
            "has_code": has_code,
            "is_math": is_math,
            "optimal_model": optimal_model,
            "rationale": rationale,
        })

    if not cleaned_rows:
        raise ValueError("Model output parsed, but no valid rows matched the schema.")

    return cleaned_rows

def deduplicate(rows: List[Dict]) -> List[Dict]:
    """
    Deduplicate the list of query rows based on the 'query' field.
    
    Args:
        rows (List[Dict]): The list of query rows to deduplicate.

    Returns:
        List[Dict]: A deduplicated list of query rows.
    """
    seen = set()
    unique_rows = []

    for row in rows:
        # normalize query 
        key = " ".join(row["query"].lower().split())
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    
    return unique_rows

def main():
    target_n = 250
    max_retries = 5
    all_rows = []
    failures = 0

    while len(all_rows) < target_n:
        if failures >= max_retries:
            print(f"Stopping after {max_retries} consecutive failures. Got {len(all_rows)} rows.")
            break
        try:
            prompt = build_generation_prompt(batch_size=25, seed=random.randint(0, 10000))
            raw = call_llm(prompt)
            batch = parse_rows(raw)
            all_rows.extend(batch)
            all_rows = deduplicate(all_rows)
            failures = 0  # reset on success
            print(f"Generated {len(all_rows)} unique rows so far...")
            pd.DataFrame(all_rows).to_csv("data/queries_in_progress.csv", index=False)  # save progress
            time.sleep(15)
        except Exception as e:
            failures += 1
            print(f"Batch failed ({failures}/{max_retries}), retrying... Error: {e}")
            continue
    
    
    df = pd.DataFrame([asdict(QueryRow(**r)) for r in all_rows[:target_n]])
    # TODO - add token_count if you want it here 

    # adding approximate token count (words * 1.3 is a reasonable estimate)
    df['token_count'] = df['query'].apply(lambda x: int(len(x.split()) * 1.3))

    df.to_csv("data/queries.csv", index=False)

if __name__ == "__main__":
    main()