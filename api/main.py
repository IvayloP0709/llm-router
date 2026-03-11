import sys 
import os 
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib 
import json 
import pandas as pd 
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from features.extractor import extract_features

# pydantic schemas 

class QueryRequest(BaseModel):
    text:str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Explain quantum entaglement in simple terms"
            }
        }

class RoutingResponse(BaseModel):
    recommended_model: str
    confidence: float 
    estimated_cost: float
    all_probabilities: dict 
    features: dict

# cost map
COST_MAP = {
    "gemini-flash": 0.001,
    "gpt-4o": 0.01,
    "o3-mini": 0.05,
}

# load model at startup

app = FastAPI(
    title ="LLM Router",
    description ="Routes queries to the optimal LLM based on cost vs quality",
    version="1.0.0"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

try: 
    model = joblib.load(os.path.join(MODEL_DIR, "router_v1.joblib"))

    with open(os.path.join(MODEL_DIR, "label_classes.json"), "r") as f:
        label_classes = json.load(f)
    
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "r") as f:
        feature_columns = json.load(f)
    
    model_loaded = True
    print(f"Model loaded. Classes: {label_classes}")
    print(f"Expected features ({len(feature_columns)}): {feature_columns[:5]}...")

except Exception as e:
    model_loaded = False
    print(f"Error loading model: {e}")

# routes

@app.get("/health")
def health():
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }

@app.post("/route", response_model=RoutingResponse)
def route_query(query: QueryRequest):
    """
    Takes a user query, extracts features, runs the XGBoost model,
    and returns which LLM to use + confidence + cost.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    text = (query.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty")
    
    # 1. extract features 
    raw_features = extract_features(text)

    # 2. align features to training column order
    feature_values = [raw_features.get(col, 0) for col in feature_columns]
    X = pd.DataFrame([feature_values], columns=feature_columns)

    # 3. predict probabilities
    pred_idx = int(model.predict(X)[0])
    probas = model.predict_proba(X)[0]

    recommended = label_classes[pred_idx]
    confidence = float(probas[pred_idx])

    # 4. build response 
    all_probs = {
        label_classes[i]: round(float(probas[i]), 4)
        for i in range(len(label_classes))
    }

    return RoutingResponse(
        recommended_model=recommended,
        confidence=round(confidence, 4),
        estimated_cost=COST_MAP.get(recommended, 0.0),
        all_probabilities=all_probs,
        features=raw_features
    )

@app.post("/route/batch")
def route_batch(queries: List[QueryRequest]) -> List[RoutingResponse]:
    """
    Route a batch of queries at once. Returns a list of routing decisions.
    """
    results = []
    
    for q in queries: 
        try:
            result = route_query(q)
            results.append(result.model_dump())
        except HTTPException as e:
            results.append({"error": f"Query failed: {e.detail}", "text": q.text})
    return results