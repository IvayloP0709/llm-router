# LLM Router

A cost-optimized routing system that automatically decides which LLM to use for any given query, balancing cost and quality. Instead of sending every query to the most expensive model, the router analyzes the query and routes it to the cheapest model that can handle it well.

```
User query → "Explain quantum entanglement in simple terms"
                         ↓
               [ Feature Extraction ]
               spaCy · tiktoken · textstat · zero-shot NLI
                         ↓
               [ XGBoost Classifier ]
                         ↓
               Route to: gemini-flash (saves 10-50x cost)
```

## Model Tiers

| Tier | Model | Use Case | Cost/Query |
|------|-------|----------|------------|
| Cheap | gemini-flash | Simple Q&A, translation, factual lookups | $0.001 |
| Mid | gpt-4o | Code generation, summaries, moderate reasoning | $0.01 |
| Expensive | o3-mini | Complex math, multi-step reasoning, analysis | $0.05 |

## Features

- **37 NLP features** extracted per query across 7 groups: structural, readability, vocabulary, POS ratios, named entities, task type (zero-shot NLI), and domain (zero-shot NLI)
- **XGBoost classifier** trained with a custom cost matrix that penalizes under-routing more than over-routing
- **FastAPI REST API** with `/route`, `/route/batch`, and `/health` endpoints
- **MLflow experiment tracking** with hyperparameter sweeps
- **Docker support** for containerized deployment
- **Render / AWS Lambda** deployment options

## Project Structure

```
llm-router/
├── data/                    # Synthetic training data generation
│   ├── generate_synthetic_data.py
│   ├── queries.csv          # 250 labeled training queries
│   └── features.csv         # Extracted features + labels
├── features/                # Feature extraction pipeline
│   ├── extractor.py         # spaCy + tiktoken + textstat + HuggingFace NLI
│   └── build_features.py    # Batch feature extraction
├── training/                # Model training & evaluation
│   ├── train.py             # Baseline XGBoost training
│   └── train_mlflow.py      # MLflow experiment tracking
├── models/                  # Trained model artifacts
│   ├── router_v1.joblib     # XGBoost model
│   ├── label_classes.json   # ["gemini-flash", "gpt-4o", "o3-mini"]
│   └── feature_columns.json # Ordered feature names
├── api/                     # FastAPI service
│   └── main.py
├── aws/                     # AWS deployment
│   └── lambda_handler.py
├── Dockerfile
├── template.yaml            # AWS SAM template
└── requirements.txt
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/llm-router.git
cd llm-router

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Usage

### Run the API locally

```bash
uvicorn api.main:app --reload
```

### Route a query

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"text": "What is photosynthesis?"}'
```

**Response:**
```json
{
  "recommended_model": "gemini-flash",
  "confidence": 0.82,
  "estimated_cost": 0.001,
  "all_probabilities": {
    "gemini-flash": 0.82,
    "gpt-4o": 0.12,
    "o3-mini": 0.06
  },
  "features": { "token_count": 5, "word_count": 4, "..." : "..." }
}
```

### Batch routing

```bash
curl -X POST http://localhost:8000/route/batch \
  -H "Content-Type: application/json" \
  -d '[{"text": "Hello world"}, {"text": "Prove the Riemann hypothesis"}]'
```

### Health check

```bash
curl http://localhost:8000/health
```

## Docker

```bash
docker build -t llm-router .
docker run -p 8000:8000 llm-router
```

## Model Performance

Best model (shallow — XGBoost, 100 trees, depth 2):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| gemini-flash | 0.68 | 1.00 | 0.81 | 19 |
| gpt-4o | 0.95 | 0.90 | 0.93 | 21 |
| o3-mini | 1.00 | 0.20 | 0.33 | 10 |
| **Accuracy** | | | **0.80** | **50** |

Trained on 250 synthetic queries generated via Gemini 2.5 Flash. 5 hyperparameter configurations tested via MLflow.

## Tech Stack

- **ML**: XGBoost, scikit-learn, pandas, numpy
- **NLP**: spaCy, tiktoken, textstat, HuggingFace Transformers (zero-shot NLI)
- **API**: FastAPI, Uvicorn, Pydantic
- **Tracking**: MLflow
- **Deployment**: Docker, AWS Lambda (Mangum)

## License

MIT
