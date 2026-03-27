from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from viet_qa.models.extractive import ExtractiveQAModel
from viet_qa.models.generative import GenerativeQAModel
from viet_qa.eval.metrics import evaluate_predictions

app = FastAPI(title="Vietnamese QA API", description="API for comparing Extractive and Generative QA on Vietnamese texts.")

# --- Schemas ---

class QARequest(BaseModel):
    context: str
    question: str

class QAResponse(BaseModel):
    answer: str
    latency_ms: float
    confidence: float
    evidence: str
    model_name: str

class CompareResponse(BaseModel):
    extractive: QAResponse
    generative: QAResponse

class EvalRequest(BaseModel):
    num_samples: int = 10
    model_type: str = "extractive"

# --- Lazy Models ---

models: Dict[str, Any] = {}

def get_model(model_type: str):
    if model_type not in models:
        try:
            if model_type == 'extractive':
                models[model_type] = ExtractiveQAModel()
            elif model_type == 'generative':
                models[model_type] = GenerativeQAModel()
            else:
                raise ValueError("Invalid model type")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return models[model_type]

# --- Endpoints ---

@app.get("/health")
def health_check():
    """Returns the API status and currently loaded models footprint."""
    return {"status": "ok", "models_loaded": list(models.keys())}

def _predict_helper(req: QARequest, model_type: str) -> QAResponse:
    """Helper method to format raw prediction dictionaries into strict schemas."""
    try:
        model = get_model(model_type)
        res = model.predict(req.question, req.context)
        
        is_supported = res.get("supported_by_context", True)
        
        if model_type == "extractive":
            evidence_str = f"Found span: '{res.get('answer', '')}'"
        else:
            evidence_str = "High Match in Context" if is_supported else "Unsupported / Hallucinated"
            
        return QAResponse(
            answer=res.get("answer", ""),
            latency_ms=res.get("latency_ms", 0.0),
            confidence=res.get("confidence", 0.0),
            evidence=evidence_str,
            model_name=res.get("model_name", "unknown")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/extractive", response_model=QAResponse)
def predict_extractive(req: QARequest):
    """Answers using the purely Extractive model."""
    return _predict_helper(req, "extractive")

@app.post("/predict/generative", response_model=QAResponse)
def predict_generative(req: QARequest):
    """Answers using the Generative model."""
    return _predict_helper(req, "generative")

@app.post("/compare", response_model=CompareResponse)
def compare_models(req: QARequest):
    """Runs BOTH extractive and generative models simultaneously for comparison."""
    ext_res = _predict_helper(req, "extractive")
    gen_res = _predict_helper(req, "generative")
    return CompareResponse(extractive=ext_res, generative=gen_res)

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    """Evaluates the model on a subset of the validation dataset."""
    from viet_qa.data.loader import load_qa_dataset, format_example
    try:
        dataset = load_qa_dataset("validation", max_samples=req.num_samples)
        model = get_model(req.model_type)
        
        predictions = []
        references = []
        latencies = []
        
        for item in dataset:
            ex = format_example(item)
            res = model.predict(ex["question"], ex["context"])
            predictions.append(res["answer"])
            references.append(ex["answers"])
            latencies.append(res["latency_ms"])
            
        metrics = evaluate_predictions(predictions, references)
        metrics["avg_latency_ms"] = sum(latencies) / len(latencies) if latencies else 0.0
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
