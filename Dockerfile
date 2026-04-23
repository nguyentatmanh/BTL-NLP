FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first for caching purposes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest httpx torchvision

# Copy all project source code
COPY . .

ENV PYTHONPATH=/app/viet_qa/src
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
