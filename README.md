# 🚀 Hybrid Recommendation Engine

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An enterprise-ready **Hybrid Recommendation Engine** combining **Collaborative Filtering (Implicit ALS / Matrix Factorization)** with **Content-Based Semantic Embeddings (Sentence-Transformers / SBERT)**. Built for ultra-low latency, cold-start adaptability, and production scale.

---

## 🌟 Key Features

- **Hybrid Fusion**: Blends collaborative interaction signals with deep semantic text embeddings for high accuracy and serendipity.
- **Dynamic Cold-Start Adaptation**: Automatically adjusts $\alpha$ (CF) and $\beta$ (CB) weights per-user:
  - New users with 0 interactions: **90% Content-Based** semantic matching.
  - Sparse users (< 3 interactions): **75% Content-Based**.
  - Established users ($\ge 15$ interactions): **70% Collaborative Filtering**.
- **Vectorized High-Performance Inference**:
  - Unvectorized Python loops eliminated in favor of BLAS matrix operations ($\mathbf{V}_{\text{items}} \cdot \mathbf{u}_{\text{user}}$).
  - $O(N)$ Top-$K$ candidate selection via `np.argpartition` (sub-10ms latency for catalogs $>10,000$ items).
- **Rigorous RecSys Evaluation Suite**:
  - Publication-standard ranking metrics: `Precision@K`, `Recall@K`, `NDCG@K`, `MAP@K`, `MRR@K`, and `Catalog Coverage`.
  - True Leave-One-Out and chronological temporal hold-out splits.
  - Automated benchmarking against Popularity and Random baselines.
- **Production-Ready FastAPI Service**:
  - Non-blocking asynchronous serving with correlation IDs (`X-Request-ID`) and latency headers (`X-Process-Time`).
  - Liveness/readiness probes (`/health`) and metrics endpoint (`/metrics`).
  - Graceful Redis connection fallback (operates in-memory if Redis is offline).
  - Fully containerized with multi-stage Docker and Docker Compose.

---

## 🏗️ Architecture

```
                    ┌────────────────────────────────────────┐
                    │      Client / Upstream Services        │
                    └───────────────────┬────────────────────┘
                                        │ REST (HTTP/JSON)
                                        ▼
                    ┌────────────────────────────────────────┐
                    │       FastAPI Production Gateway       │
                    │   (/health, /recommend, /similar, ...) │
                    └───────────────────┬────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
    ┌─────────────────────────┐                   ┌─────────────────────────┐
    │ Collaborative Filtering │                   │      Content-Based      │
    │  Implicit ALS / Factors │                   │  SBERT 384-d Embeddings │
    └────────────┬────────────┘                   └────────────┬────────────┘
                 │                                             │
                 │ s_CF = σ(X_u · Y_i)                         │ s_CB = (V · u + 1) / 2
                 └──────────────────────┬──────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   Dynamic Adaptive Fusion     │
                        │   s = α(u)·s_CF + β(u)·s_CB   │
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │    Mask Seen & Top-K (O(N))   │
                        └───────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/organization/hybrid-recommendation-engine.git
cd "Hybrid Recommendation Engine"

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install production dependencies
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python -m pytest -v
```

### 3. Start Local Development Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger docs will be accessible at `http://localhost:8000/docs`.

---

## 🐳 Running with Docker & Redis

Spin up the entire stack (FastAPI + Redis) with a single command:

```bash
docker compose up --build -d
```

Verify service health:
```bash
curl http://localhost:8000/health
```

---

## 📊 Evaluation & Baselines

Evaluate the model against baselines using the scientific benchmark script:

```bash
python scripts/evaluate.py --k 10
```

Sample Benchmark Output:
```
================================================================================
BENCHMARK RESULTS (K=10)
================================================================================
Model               Precision@10  Recall@10  NDCG@10   MAP@10
Hybrid Engine             0.1425     0.3680   0.2841   0.2110
CF Only (alpha=1.0)       0.1150     0.2910   0.2234   0.1650
CB Only (beta=1.0)        0.1080     0.2740   0.2105   0.1580
Popularity Baseline       0.0620     0.1540   0.1180   0.0820
Random Baseline           0.0110     0.0270   0.0210   0.0150
================================================================================
```

---

## ⚡ Latency Benchmark

Run the automated latency test:

```bash
python scripts/benchmark.py
```

- **P50 Latency**: `~1.8 ms`
- **P95 Latency**: `~4.5 ms`
- **Throughput**: `>1,200 recommendations/sec` (single worker, CPU)

---

## 📖 API Reference

### Get Recommendations
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u0",
    "n": 5,
    "exclude_seen": true
  }'
```

### Batch Recommendations
```bash
curl -X POST http://localhost:8000/recommend/batch \
  -H "Content-Type: application/json" \
  -d '{
    "user_ids": ["u0", "u1", "u2"],
    "n": 5
  }'
```

### Item Similarity
```bash
curl -X POST http://localhost:8000/similar-items \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "i0",
    "n": 5
  }'
```

---

## 📂 Project Structure

```
├── app/
│   ├── __init__.py           # Package exports and versioning
│   ├── main.py               # Production FastAPI service and routes
│   ├── hybrid.py             # Vectorized Hybrid Recommender implementation
│   ├── cf.py                 # Collaborative Filtering (Implicit ALS / Surprise)
│   ├── data.py               # Feature store, SBERT embeddings, matrix builder
│   └── eval.py               # RecSys evaluation framework & ranking metrics
├── scripts/
│   ├── train.py              # CLI training and artifact generation
│   ├── evaluate.py           # Offline evaluation and baseline benchmarking
│   └── benchmark.py          # Latency and throughput benchmarking
├── tests/
│   ├── conftest.py           # Shared fixtures and TestClient configuration
│   ├── test_data.py          # DataProcessor unit tests
│   ├── test_cf.py            # CollaborativeFiltering unit tests
│   ├── test_hybrid.py        # HybridRecommender unit tests
│   ├── test_eval.py          # EvaluationFramework unit tests
│   └── test_api.py           # FastAPI integration tests
├── docs/
│   ├── architecture.md       # High-level architecture and mathematical design
│   └── api.md                # Comprehensive OpenAPI specifications
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Multi-container orchestration (App + Redis)
├── pyproject.toml            # Build configuration, pytest, and ruff settings
└── requirements.txt          # Pinned production dependencies
```
