from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from viet_qa.models.extractive import ExtractiveQAModel
from viet_qa.models.generative import GenerativeQAModel
from viet_qa.models.retriever import TfidfRetriever
from viet_qa.eval.metrics import evaluate_predictions

# --- Global state ---
models: Dict[str, Any] = {}
retriever = TfidfRetriever()
dataset_contexts: List[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dataset and build TF-IDF index on startup."""
    global dataset_contexts
    print("Loading dataset and building TF-IDF index...")
    try:
        from viet_qa.data.loader import load_qa_dataset
        dataset = load_qa_dataset("all")
        seen = set()
        for split_name in dataset.keys():
            for item in dataset[split_name]:
                ctx = item.get("context", "")
                if ctx and ctx not in seen:
                    dataset_contexts.append(ctx)
                    seen.add(ctx)
        print(f"  Total unique contexts: {len(dataset_contexts)}")
        retriever.build_index(dataset_contexts)
    except Exception as e:
        print(f"Warning: Failed to load dataset: {e}")
    yield


app = FastAPI(
    title="Vietnamese QA API",
    description="Open-domain QA: Retriever (TF-IDF) + Reader (Extractive QA Transformer).",
    lifespan=lifespan,
)

# --- Schemas ---

class QARequest(BaseModel):
    context: str
    question: str

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    model_type: str = "extractive"

class CandidateResult(BaseModel):
    rank: int
    answer: str
    final_score: float
    reader_score: float
    retriever_score: float
    context_snippet: str
    full_context: str = ""

class AskResponse(BaseModel):
    answer: str
    final_score: float
    status: str
    best_context: str
    candidates: List[CandidateResult]

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
    return {
        "status": "ok",
        "models_loaded": list(models.keys()),
        "retriever_ready": retriever.is_built,
        "total_contexts": len(dataset_contexts),
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Open-domain QA: chỉ cần câu hỏi.
    Retriever tìm top-k context → Reader trích xuất đáp án → trả kết quả tốt nhất.
    """
    if not retriever.is_built:
        raise HTTPException(status_code=503, detail="Retriever chưa sẵn sàng.")

    search_results = retriever.search(req.question, top_k=req.top_k)
    model = get_model(req.model_type)

    candidates = []
    for rank, (idx, ret_score, context) in enumerate(search_results, 1):
        try:
            res = model.predict(req.question, context)
            reader_score = res.get("confidence", 0.0)
            
            # Khắc phục sai sót: Reader thường hay "tự tin thái quá" vào ngữ cảnh sai.
            # Dùng trọng số 60% cho bài toán tìm kiếm (Retriever) và 40% cho người đọc (Reader).
            final_score = 0.6 * ret_score + 0.4 * reader_score
            
            candidates.append(CandidateResult(
                rank=rank,
                answer=res.get("answer", ""),
                final_score=round(final_score, 3),
                reader_score=round(reader_score, 3),
                retriever_score=round(ret_score, 3),
                context_snippet=context[:200] + "..." if len(context) > 200 else context,
                full_context=context,
            ))
        except Exception:
            candidates.append(CandidateResult(
                rank=rank, answer="Error", final_score=0.0,
                reader_score=0.0, retriever_score=round(ret_score, 3),
                context_snippet=context[:200],
                full_context=context,
            ))

    # Sort by final_score descending
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    # Re-rank
    for i, c in enumerate(candidates):
        c.rank = i + 1

    best = candidates[0] if candidates else None
    if best and best.final_score > 0.1:
        status = "OK"
    else:
        status = "Low confidence"

    return AskResponse(
        answer=best.answer if best else "Không tìm thấy",
        final_score=best.final_score if best else 0.0,
        status=status,
        best_context=best.full_context if best else "",
        candidates=candidates,
    )


def _predict_helper(req: QARequest, model_type: str) -> QAResponse:
    try:
        model = get_model(model_type)
        res = model.predict(req.question, req.context)
        is_supported = res.get("supported_by_context", True)
        if model_type == "extractive":
            evidence_str = f"Found span: '{res.get('answer', '')}'"
        else:
            evidence_str = "High Match" if is_supported else "Unsupported"
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
    return _predict_helper(req, "extractive")

@app.post("/predict/generative", response_model=QAResponse)
def predict_generative(req: QARequest):
    return _predict_helper(req, "generative")

@app.post("/compare", response_model=CompareResponse)
def compare_models(req: QARequest):
    ext_res = _predict_helper(req, "extractive")
    gen_res = _predict_helper(req, "generative")
    return CompareResponse(extractive=ext_res, generative=gen_res)

@app.post("/evaluate")
def evaluate(req: EvalRequest):
    from viet_qa.data.loader import load_qa_dataset, format_example
    try:
        dataset = load_qa_dataset("validation", max_samples=req.num_samples)
        model = get_model(req.model_type)
        predictions, references, latencies = [], [], []
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
