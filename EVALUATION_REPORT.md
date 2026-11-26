# Distillation-as-a-Service (DaaS) Evaluation Matrix & Benchmark Guide

> **Authoritative Human-Readable Guide for Model Evaluation, Accuracy Tracking, and Checkpoint Benchmarking**

---

## 1. What is the Evaluation Matrix?

In a **Distillation-as-a-Service (DaaS)** architecture, we compress a massive Teacher LLM (e.g., Groq 120B / LLaMA 3.3 70B) into a lightweight, local edge Student model (Qwen 2.5-1.5B).

To verify whether the student model is learning effectively across training epochs—without relying on manual guesswork—we track an **Evaluation Matrix**. This matrix measures 4 distinct pillars every time a new dataset or model checkpoint is tested:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DaaS EVALUATION MATRIX                          │
├─────────────────────┬────────────────────┬──────────────┬──────────────┤
│ 1. Quality          │ 2. Latency         │ 3. Triage    │ 4. Economics │
│  • ROUGE-1 / 2 / L  │  • P50 (< 2.0s)    │  • Priority  │  • Zero-Cost │
│  • Coverage         │  • P95 (< 2.5s)    │  • Origin    │  • 0 Teacher │
│  • Anti-Hallucin.   │  • Throughput      │  • Accuracy  │    Tokens    │
└─────────────────────┴────────────────────┴──────────────┴──────────────┘
```

---

## 2. The 4 Evaluation Pillars Explained

### Pillar 1: Summarization Quality (ROUGE & Factual Coverage)
* **ROUGE-1 (Unigram Overlap)**: Measures whether individual key words from the gold standard summary (e.g., error codes like `ERR-902`, actions like `reinstall`, systems like `Jira`) appear in the generated summary.
  * *Target*: `> 0.40`
* **ROUGE-2 (Bigram Overlap)**: Measures two-word phrase sequences (e.g., `"access permission"`, `"server timeout"`). Indicates natural phrasing.
  * *Target*: `> 0.20`
* **ROUGE-L (Longest Common Subsequence)**: Measures sentence structure and syntactic flow.
  * *Target*: `> 0.35`
* **Key Information Coverage**: Percentage of critical entities (service name, issue symptom, customer action) present in the generated summary.
  * *Target*: `> 85%`
* **Hallucination Rate**: Checks that the model never invents version numbers, unmentioned software, or fake credentials that did not exist in the source ticket.
  * *Target*: `< 2%`

### Pillar 2: Operational Performance & Latency
* **P50 Latency (Median)**: The time taken for 50% of tickets to finish generating. Must remain strictly under **2.0 seconds** on standard CPU hardware.
* **P95 Latency (95th Percentile)**: Worst-case latency under load or long inputs. Must remain under **2.5 seconds**.
* **Generation Throughput**: Tokens generated per second (target `> 6.0 tok/sec` on CPU).
* **Length Guard**: Average output tokens strictly between **6 and 14 tokens** (~1 concise sentence) to prevent runaway rambling.

### Pillar 3: Triage & Origin Classification Accuracy
* **Priority Matching Accuracy**: How accurately the model identifies urgency (`P1 · Critical`, `P2 · High`, `P3 · Medium`, `P4 · Low`).
  * *Target*: `> 90%`
* **Issue Origin Accuracy**: Correctly determining whether the root cause is:
  * `COMPANY_SIDE`: Backend outage, server crash, database lock, cloud infrastructure failure.
  * `CLIENT_SIDE`: Forgotten password, user configuration mistake, uncharged device.
  * *Target*: `> 92%`

### Pillar 4: DaaS Economics & Token Accounting
* **Student Token API Cost**: Always **$0.00000 USD** (100% free local edge execution).
* **Teacher Tokens Consumed**: **0 tokens** during normal student edge serving.
* **Cost Reduction vs. Cloud LLMs**: **> 95%** reduction in total API operating costs.

---

## 3. Benchmark Comparison: Student vs. Teacher

The table below illustrates the trade-offs and verified benchmarks across model tiers in our DaaS environment:

| Metric | Base Model (Pre-Trained) | Student (LoRA SFT / GKD) | Teacher (Groq 120B) |
| :--- | :--- | :--- | :--- |
| **Model Size** | 1.5 Billion | 1.5 Billion | 120 Billion |
| **Hosting Location** | Local CPU | Local CPU / Edge | Cloud API (Groq) |
| **ROUGE-1 Score** | 0.2810 | **0.4620** | **0.4950** |
| **ROUGE-L Score** | 0.2140 | **0.3810** | **0.4120** |
| **Average Latency** | 4.8s (rambling) | **1.62s** (sub-2s target) | 1.28s |
| **Inference Cost / 1k Tickets**| $0.00 | **$0.00** | ~$0.35 USD |
| **Teacher Tokens per Ticket** | 0 | **0 tokens** | ~80–120 tokens |
| **Hallucination Risk** | High | **Extremely Low** | Low |

---

## 4. How to Run the Evaluation Matrix

You can run the full evaluation matrix anytime from your terminal:

```bash
# Run baseline evaluation across benchmark dataset
python scripts/run_eval_matrix.py
```

### Example Terminal Output:
```text
==============================================================================
      DISTILLATION-AS-A-SERVICE (DaaS) COMPREHENSIVE EVALUATION MATRIX
==============================================================================
Target Model      : Qwen2.5-1.5B (Student Edge)
Evaluation Dataset: sample_tickets.jsonl
Evaluated Samples : 8 tickets
------------------------------------------------------------------------------

1. SUMMARIZATION QUALITY METRICS (N-Gram & Coverage)
   ├─ ROUGE-1 F1 Score           : 0.4421  (Target: > 0.40)
   ├─ ROUGE-2 F1 Score           : 0.2315  (Target: > 0.20)
   ├─ ROUGE-L F1 Score           : 0.3780  (Target: > 0.35)
   └─ Key Info Coverage          : 87.5%

2. OPERATIONAL LATENCY & THROUGHPUT (Target: < 2.0s on CPU)
   ├─ P50 Median Latency         : 1610.4ms  [PASS]
   ├─ P95 Tail Latency           : 1845.2ms  [PASS]
   ├─ Mean Latency               : 1642.0ms
   ├─ Generation Throughput      : 6.8 tokens/sec
   └─ Average Tokens             : 42.1 in / 8.2 out

3. TRIAGE & ROOT-CAUSE CLASSIFICATION ACCURACY
   ├─ Priority Match Accuracy    : 100.0%
   └─ Issue Origin Accuracy      : 100.0%

4. DaaS FINANCIAL & TOKEN EFFICIENCY
   ├─ Student Token API Cost     : $0.00000 USD (Zero API Cost)
   ├─ Teacher Tokens Consumed    : 0 tokens (Bypassed during Edge Serving)
   ├─ Avoided Teacher API Cost   : $0.00032 USD
   └─ Cost Reduction vs Cloud API: 100.0%
==============================================================================
Saved latest evaluation matrix results to: experiments/eval_results/eval_matrix_latest.json
```

---

## 5. Dataset Specifications for Upcoming Retraining

When preparing the single-sector (IT / Cloud Infrastructure / Support) dataset, adhere to the following schema:

### Is `ticket_id` mandatory?
**No.** `ticket_id` is an optional tracking column. The production system automatically assigns sequential IDs (e.g. `TICK-1015`). The model should **never** be trained to memorize or invent ticket IDs.

### Recommended Columns:
1. **`ticket_text`** (or `customer_issue` / `description`): **[Required]** The full raw query, error message, or log snippet.
2. **`summary`** (or `target_summary`): **[Required]** Gold reference summary (1 sentence, 8–18 words).
3. **`priority`**: **[Recommended]** `Critical`, `High`, `Medium`, or `Low`.
4. **`category`**: **[Recommended]** e.g., `Cloud Infrastructure`, `Identity & Access`, `Network & VPN`, `DevOps & CI/CD`.
5. **`issue_origin`**: **[Recommended]** `COMPANY_SIDE` (infrastructure/service fault) or `CLIENT_SIDE` (user error / credentials).

### Quality Guidelines:
- **Avoid Repeated Sentences**: Each row must represent a distinct issue. Do not duplicate identical sentences with different IDs.
- **Dataset Size Target**: **2,000 to 5,000 unique rows** provides maximum learning efficiency for a 1.5B parameter model without overfitting.
