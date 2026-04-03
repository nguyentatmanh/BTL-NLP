# Vietnamese Question Answering Benchmark

A lightweight, robust and production-like scaffolding project for comparing Extractive QA vs Generative QA (RAG/LLM) on Vietnamese texts.

## Architecture
- **Backend**: FastAPI for inference and evaluation endpoints.
- **Frontend**: Streamlit UI for side-by-side comparison of QA models.
- **Models**:
  - Extractive QA: e.g., `thewolfstar1/qa-vietnamese-model` or `nguyenvulebinh/vi-mrc-base`
  - Generative QA: e.g., `Qwen/Qwen2.5-1.5B-Instruct`
- **Dataset**: `ntphuc149/ViSpanExtractQA` from Hugging Face.
- **Evaluation**: Exact Match (EM), Macro F1, and Latency tracking.

## Usage Instructions

### 1. Setup Environment

Create a virtual environment and install dependencies:

```bash
cd "d:/BTL NLP/viet_qa"
python -m venv venv
# Activate the virtual environment:
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the FastAPI Backend

The backend exposes prediction and evaluation endpoints.

```bash
# Run from the project root directory
uvicorn src.viet_qa.api.main:app --reload --port 8000
```
Swagger UI will be available at: http://localhost:8000/docs

### 3. Run the Streamlit UI

In a new terminal (with the virtual environment activated), start the UI:

```bash
# Set PYTHONPATH if needed, usually running from root is sufficient
streamlit run src/viet_qa/ui/app.py
```
The UI typically starts on http://localhost:8501

### 4. Training Extractive QA Model Locally

You can fine-tune your own extractive QA model (e.g. PhoBERT) using our training pipeline. The configuration holds the hyperparameters (batch size, learning rate, checkpoint location). See `src/viet_qa/config/train_config.py`.

```bash
# Set PYTHONPATH so the imports work correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Windows equivalent:
# set PYTHONPATH=%cd%\src

# Run the training script (downloads dataset, prepares sliding-window features, and trains)
python -m viet_qa.train.train_extractive
```

### 5. Evaluating Extractive QA Checkpoints

After training, you can evaluate your specific checkpoint.

```bash
# Evaluate the final trained checkpoint (or any HuggingFace Repo ID)
python -m viet_qa.train.eval_extractive --model_path "checkpoints/extractive" --samples 500
```

### 6. Docker & Testing

Run the entire suite seamlessly isolated via Docker Compose:
```bash
docker-compose up --build
```
This automatically spins up the API backend on port `8000` and the Streamlit UI on port `8501`. 

To run the unit and smoke tests locally:
```bash
pip install pytest httpx
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Windows equivalent: set PYTHONPATH=%cd%\src
pytest tests/
```

