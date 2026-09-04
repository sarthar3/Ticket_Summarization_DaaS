# Ticket Summarization DaaS (Data-as-a-Service)

A production-ready, highly modular system for summarizing support/customer tickets using small Student LLMs.

---

## Key Features (Phase 1 Baseline)
- **Configurable Student Model**: Dynamic loading via Hugging Face (`Qwen/Qwen2.5-1.5B-Instruct`, `meta-llama/Llama-3.2-1B-Instruct`, or custom models) via environment variables or YAML configs.
- **End-to-End Inference Pipeline**: Request validation, text normalization, prompt templating, tokenization, generation, post-processing, and performance timing.
- **Token Statistics & Context Analysis**: Dataset analysis reporting Min, Max, Mean, P50, P95, and context overflow percentages.
- **Modular Evaluation Framework**: Measures ROUGE-L, key-information coverage, latency percentiles (P50, P95, mean), throughput (tokens/sec), VRAM memory usage, and estimated hardware inference costs.
- **FastAPI Production Service**: Clean REST API providing `GET /health` and `POST /summarize` with privacy-preserving logging (ticket text is kept private by default).
- **Reproducible Baseline Experiments**: CLI script (`experiments/run_baseline.py`) for benchmarking datasets without starting the API server.

---

## Project Structure

```
Tiket summarization DaaS/
├── app/
│   ├── api/
│   │   ├── routes.py          # FastAPI endpoints (GET /health, POST /summarize)
│   │   └── schemas.py         # Pydantic schemas
│   ├── config/
│   │   └── settings.py        # Centralized config loader (YAML + Env variables)
│   ├── evaluation/
│   │   ├── metrics.py         # ROUGE-L, Latency P50/P95, Throughput & Cost metrics
│   │   └── evaluator.py       # Aggregate evaluation report generator
│   ├── inference/
│   │   ├── student_model.py   # Hugging Face Causal LM model wrapper
│   │   └── pipeline.py        # End-to-end inference flow
│   ├── preprocessing/
│   │   ├── cleaner.py         # Text cleaning, normalization, validation
│   │   └── token_stats.py     # Token length distribution analyzer
│   ├── utils/
│   │   └── logger.py          # Privacy-aware JSON logger
│   └── main.py                # FastAPI web app entrypoint
├── configs/
│   └── default_config.yaml    # Default model, context limits, device, precision, seed
├── data/
│   └── sample/
│       └── sample_tickets.jsonl  # Fixed sample dataset
├── experiments/
│   ├── results/               # Generated evaluation reports (JSON)
│   └── run_baseline.py        # CLI baseline experiment runner
├── scripts/
│   └── generate_sample_data.py # Sample data generator
├── tests/                     # Unit test suite
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quickstart Guide

### 1. Installation
Clone the repository and install requirements in a virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration & Environment Overrides
Edit `configs/default_config.yaml` or set environment variables to change models or precision without modifying code:

```bash
# Example environment variable overrides:
export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export PRECISION="float16"
export DEVICE="cuda" # or "cpu" / "auto"
export RANDOM_SEED=42
```

---

## Running the API Server

Start the FastAPI production server using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Example API Requests

#### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "precision": "float16",
  "device": "auto"
}
```

#### 2. Summarize Ticket
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "T001",
    "ticket_text": "Customer reported being unable to login to their mobile banking app after the v4.2 update on iOS 17.4. Password reset failed. Needs urgent payroll access."
  }'
```

**Structured Response:**
```json
{
  "ticket_id": "T001",
  "summary": "User unable to login to mobile banking app following v4.2 update on iOS. Password reset failed; needs urgent payroll access.",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "latency_ms": 142.50,
  "input_tokens": 42,
  "output_tokens": 28
}
```

---

## Running Baseline Evaluation

Execute the CLI baseline script against the test dataset to compute token length statistics, model quality, performance percentiles, and cost estimates:

```bash
python experiments/run_baseline.py --config configs/default_config.yaml --dataset data/sample/sample_tickets.jsonl
```

### Sample Output Metrics Report
```text
============================================================
            BASELINE EXPERIMENT REPORT
============================================================
Model Name           : Qwen/Qwen2.5-1.5B-Instruct
Evaluated Samples    : 8

[QUALITY METRICS]
  ROUGE-1            : 0.5842
  ROUGE-2            : 0.3120
  ROUGE-L            : 0.5215
  Key Info Coverage  : 0.8125

[PERFORMANCE METRICS]
  P50 Latency        : 135.20 ms
  P95 Latency        : 210.40 ms
  Mean Latency       : 148.10 ms
  Throughput         : 195.40 tokens/sec
  Avg Input Tokens   : 48.50
  Avg Output Tokens  : 26.10

[ESTIMATED INFERENCE COST]
  Cost / Ticket      : $0.000035
  Cost / 1k Tickets  : $0.0350
============================================================
```

---

## Running Unit Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

---

## Future Phase Roadmap
- **Phase 2**: Baseline Evaluation on full domain dataset splits.
- **Phase 3**: Supervised Fine-Tuning (SFT) with LoRA / QLoRA.
- **Phase 4**: Failure-case analysis & automated hallucination auditing.
- **Phase 5 & 6**: Teacher LLM Integration & Generalized Knowledge Distillation (GKD).
- **Phase 7**: vLLM deployment optimization, AWS g6.2xlarge L4 scaling, and dynamic batching.
