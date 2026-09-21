# Production Ticket Summarization DaaS (Data-as-a-Service)

A production-oriented, end-to-end ML system for summarizing customer and support tickets using a Student LLM baseline (`Qwen/Qwen2.5-1.5B-Instruct` / `Llama-3.2-1B`), Supervised Fine-Tuning (SFT), Teacher LLM Supervision, and Generalized Knowledge Distillation (GKD).

---

## Complete Development Roadmap (All 7 Phases Implemented)

- **Phase 1: Student Model Baseline & API Pipeline**: Dynamic model loading (`MODEL_NAME` env override), text cleaning, prompt formatting, token stats calculator, FastAPI (`GET /health`, `POST /summarize`), privacy-aware logging.
- **Phase 2: Baseline Evaluation & Immutable Test Set**: Deterministic dataset partitioner (`DatasetSplitter`, seed=42) generating fixed `train`, `val`, and `test.jsonl` splits in `data/processed/`, evaluation suite computing ROUGE-1/2/L, key coverage, P50/P95/mean latency, throughput, VRAM, and cost.
- **Phase 3: Supervised Fine-Tuning (SFT)**: `StudentSFTTrainer` with PEFT LoRA adapters (`r=16`, `lora_alpha=32`), loss logging, and adapter checkpoint saving to `checkpoints/sft_student/`.
- **Phase 4: Failure-Case Analysis**: Automated error detection engine (`FailureAnalyzer`) auditing 5 defect modes (hallucinations, omissions, repetition loops, truncation, length defects) saving diagnostic reports (`failure_analysis_report.json`).
- **Phase 5: Teacher LLM Integration**: Extensible provider abstraction (`OpenAITeacher` `gpt-4o`, `AnthropicTeacher` `claude-3-5-sonnet`, `MockTeacher`), secret-masked API key handling, and gold supervision label generator (`train_teacher_distill.jsonl`).
- **Phase 6: Generalized Knowledge Distillation (GKD)**: `GKDSFTTrainer` distilling Teacher supervision targets into the Student Model with LoRA adapters and weighted distillation loss (`distill_alpha=0.6`), saving `checkpoints/gkd_student/`.
- **Phase 7: Deployment Optimization & Scaling**: Production configuration (`configs/deployment.yaml`), quantization engine (`FP16/INT8/INT4`), multi-stage Docker build (`Dockerfile`), `docker-compose.yml`, operational health & `/metrics` endpoints.

---

## Project Structure

```
Tiket summarization DaaS/
├── app/
│   ├── api/
│   │   ├── routes.py          # FastAPI endpoints (GET /health, GET /metrics, POST /summarize)
│   │   └── schemas.py         # Pydantic schemas
│   ├── config/
│   │   └── settings.py        # Centralized config loader (YAML + Env variables)
│   ├── evaluation/
│   │   ├── metrics.py         # ROUGE-L, Latency P50/P95, Throughput & Cost metrics
│   │   ├── evaluator.py       # Aggregate evaluation report generator
│   │   ├── tracker.py         # Experiment tracker logger
│   │   └── failure_analysis.py# Automated failure case detection engine
│   ├── inference/
│   │   ├── student_model.py   # Hugging Face Causal LM model wrapper
│   │   ├── pipeline.py        # End-to-end inference execution flow
│   │   └── quantization.py    # Quantization (FP16, BF16, INT8, INT4) config manager
│   ├── preprocessing/
│   │   ├── cleaner.py         # Text cleaning, normalization, validation
│   │   ├── token_stats.py     # Token length distribution analyzer
│   │   └── dataset.py         # Deterministic dataset partitioner
│   ├── teacher/
│   │   └── teacher_llm.py     # OpenAI / Anthropic / Mock Teacher LLM provider
│   ├── training/
│   │   ├── sft_trainer.py     # Supervised Fine-Tuning (SFT) engine with LoRA
│   │   └── distill_trainer.py # Generalized Knowledge Distillation (GKD) engine
│   ├── utils/
│   │   └── logger.py          # Privacy-aware JSON logger
│   └── main.py                # FastAPI web app entrypoint
├── checkpoints/               # Model checkpoints & LoRA adapters
│   ├── sft_student/
│   └── gkd_student/
├── configs/
│   ├── default_config.yaml    # Default settings
│   ├── sft_config.yaml        # SFT training hyperparameters
│   ├── teacher_config.yaml    # Teacher LLM provider config
│   ├── gkd_config.yaml        # GKD distillation hyperparameters
│   └── deployment.yaml       # Deployment & quantization config
├── data/
│   ├── processed/             # Fixed dataset splits (train, val, test, distill)
│   └── sample/                # Sample tickets
├── experiments/               # Benchmark evaluation & experiment scripts
├── scripts/                   # CLI scripts for dataset prep, training, teacher labels
├── tests/                     # 27 automated unit tests
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Quickstart Guide

### 1. Installation
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Variables & Model Selection
Model selection is 100% configurable without source code modification:

```bash
export MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
export PRECISION="float16"
export DEVICE="cuda" # or "cpu" / "auto"
export RANDOM_SEED=42
```

---

## Running the API Service

### Option A: Local Uvicorn
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option B: Docker Container
```bash
docker-compose up --build -d
```

### API Endpoints Overview

#### 1. Health Readiness Probe
```bash
curl -X GET http://localhost:8000/health
```

#### 2. Operational Metrics Probe
```bash
curl -X GET http://localhost:8000/metrics
```

#### 3. Summarize Ticket Endpoint
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "T001",
    "ticket_text": "Customer reported being unable to login to mobile banking app following v4.2 update on iOS 17.4. Password reset failed. Needs urgent payroll access."
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

## Running Experiments & Pipelines

```bash
# 1. Prepare fixed train/val/test splits & print token statistics
python scripts/prepare_dataset.py

# 2. Run Phase 2 Baseline Evaluation on fixed test set
python experiments/evaluate_phase2_baseline.py

# 3. Train SFT Student Model with PEFT LoRA
python scripts/train_sft.py --config configs/sft_config.yaml

# 4. Evaluate SFT Model on fixed test set
python experiments/evaluate_phase3_sft.py

# 5. Run Automated Failure-Case Analysis
python experiments/analyze_failures.py

# 6. Generate Teacher LLM Distillation Labels
python scripts/generate_teacher_labels.py --provider mock

# 7. Evaluate Teacher Upper-Bound Quality
python experiments/evaluate_teacher.py --provider mock

# 8. Train GKD Distilled Student Model
python scripts/train_gkd.py --config configs/gkd_config.yaml

# 9. Evaluate GKD Model on fixed test set
python experiments/evaluate_phase6_gkd.py

# 10. Run Full Unit Test Suite (27 tests)
python -m pytest tests/ -v
```
