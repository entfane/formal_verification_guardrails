# Beyond Red-Teaming: Formal Guarantees of LLM Guardrail Classifiers

This repository contains the official anonymized code and resources for the NeurIPS 2026 paper submission: *"Beyond Red-Teaming: Formal Guarantees of LLM Guardrail Classifiers"*. 

## Overview
Guardrail Classifiers are used to defend production language models against harmful behavior, but their reliability has historically been heuristic, offering no formal guarantees. Standard verification methods struggle to scale due to the discrete nature of linguistic input space and the non-linear depth of Transformer architectures. 

This repository provides a framework that closes this gap by shifting verification from the discrete input space to the classifier's pre-activation space. By leveraging the monotonicity of the sigmoid classification head, we certify semantic regions of harmful behavior without approximation in $O(d)$ time. 

## Key Contributions
*   **Closed-Form Gap-Analysis Framework:** We provide a linear verification procedure that isolates the classification head on the last hidden state, avoiding the need to propagate bounds through the entire Transformer stack.
*   **Complementary Specifications of Harm:** The codebase implements three geometric constructions of safety specifications:
    *   **Single SVD-aligned Hyper-rectangle:** Yielding deterministic, exact SAT/UNSAT certificates over a single, unified bounded region.
    *   **Multiple SVD-aligned Hyper-rectangles:** Utilizing HDBSCAN clustering to partition the space, yielding exact SAT/UNSAT certificates for localized sub-clusters of harm.
    *   **Gaussian Mixture Models (GMM):** Yielding probabilistic certificates over semantically coherent clusters.
*   **Empirical Safety Gap Analysis:** The framework exposes verifiable safety holes across three author-trained Guardrail Classifiers (BERT, GPT-2, and Llama-3.1-8B), demonstrating that despite high empirical metrics, these models lack deterministic safety guarantees under both optimal ($\tau^*$) and pessimistic ($\tau_{pess}$) thresholds.

---

## Installation & Setup
```bash
pip install -r requirements.txt
```

## ⚠️ Data Requirements for Verification
The verification scripts (`src/deterministic_verification.py` and `src/verify_gmm.py`) are designed to certify a **specification of harm**. 

To ensure valid results, the dataset provided via the `--dataset` argument for these two scripts **must contain only harmful samples**. These samples should align with the specific definition of "harm" established in your model's constitution (e.g., specific categories of bias, toxicity, or PII leakage).
**Evaluation Scripts:** Note that the scripts in `src/evaluations/` *do* require a binary labeled dataset (both harmful and harmless) to calculate Precision/Recall.

## Execution Commands

### 1. Deterministic Verification — `src/deterministic_verification.py`

Constructs a geometric specification of harm in the pre-activation space and issues an exact SAT/UNSAT certificate: **UNSAT** means every point within the specified region is classified above the threshold; **SAT** means a counterexample exists — a point inside the region that falls below it.

The script supports two specification modes controlled by `--use-single-hyper-rectangle`:

**Single SVD-aligned hyper-rectangle** (e.g. for BERT):
```bash
python src/deterministic_verification.py \
    --model "bert-guardrail" \
    --pooling "first" \
    --dataset "your-dataset.jsonl" \
    --split "train" \
    --input-col "prompt" \
    --threshold 0.48 \
    --use-single-hyper-rectangle true
```

**Multiple SVD-aligned hyper-rectangles via HDBSCAN** (e.g. for GPT-2):
```bash
python src/deterministic_verification.py \
    --model "gpt2-guardrail" \
    --pooling "last" \
    --dataset "your-dataset.jsonl" \
    --split "train" \
    --input-col "prompt" \
    --threshold 0.37 \
    --use-single-hyper-rectangle false \
    --min-cluster 25
```

| Argument | Default | Description |
|---|---|---|
| `--model` / `-m` | required | HuggingFace model ID or local path |
| `--pooling` / `-p` | required | `first` (encoder/CLS) or `last` (decoder) |
| `--dataset` / `-d` | required | HuggingFace dataset ID or path to `.jsonl` |
| `--split` / `-s` | required | Dataset split (e.g. `train`) |
| `--threshold` / `-t` | required | Classification threshold in $(0, 1)$ (e.g. `0.5`) |
| `--input-col` / `-i` | required | Column containing user input text |
| `--output-col` / `-o` | `None` | Optional assistant output column |
| `--use-single-hyper-rectangle` | `true` | `true` for a single unified rectangle; `false` for HDBSCAN clusters |
| `--min-cluster` | `5` | Minimum cluster size for HDBSCAN (only used when `--use-single-hyper-rectangle false`) |
| `--batch-size` / `-b` | `2` | Batch size for embedding extraction |
| `--max-len` / `-l` | `128` | Max tokenizer sequence length |

**Output:**
```
UNSAT
Everything inside the hyper-rectangle classified > 0.48
```
or
```
SAT
There exists a point within the hyper-rectangle which is classified <= 0.48
```

---

### 2. Probabilistic Verification — `src/verify_gmm.py`

Fits a Gaussian Mixture Model to the pre-activation embeddings of harmful inputs, then analytically derives the total probability mass $P(\hat{y} > \tau)$ by propagating each Gaussian component through the linear classification head in closed form. No sampling is required.

```bash
python src/verify_gmm.py \
    --model "gpt2-guardrail" \
    --pooling "last" \
    --dataset "your-dataset.jsonl" \
    --split "train" \
    --input-col "prompt" \
    --threshold 0.37 \
    --n-components 3 \
    --cov-type full
```

| Argument | Default | Description |
|---|---|---|
| `--model` / `-m` | required | HuggingFace model ID or local path |
| `--pooling` / `-p` | required | `first` (encoder/CLS) or `last` (decoder) |
| `--dataset` / `-d` | required | HuggingFace dataset ID or path to `.jsonl` |
| `--split` / `-s` | required | Dataset split (e.g. `train`) |
| `--threshold` / `-t` | required | Classification threshold in $(0, 1)$ (e.g. `0.5`) |
| `--input-col` / `-i` | required | Column containing user input text |
| `--output-col` / `-o` | `None` | Optional assistant output column; enables chat-template formatting when provided |
| `--n-components` / `-n` | `2` | Number of GMM mixture components |
| `--cov-type` | `full` | GMM covariance structure: `full`, `tied`, `diag`, or `spherical` |
| `--batch-size` / `-b` | `2` | Batch size for embedding extraction |
| `--max-len` / `-l` | `128` | Max tokenizer sequence length |

**Output:**
```
P(score > 0.37) = 0.90
```

---

### 3. Shape Evaluation Scripts

The `src/evaluations/` directory contains standalone scripts that assess the **geometric quality** of each specification type — independent of the classifier threshold. Each script reserves a held-out evaluation split (100 harmful + 100 harmless samples by default), constructs its geometric region on the remaining harmful-only construction data, and reports Precision / Recall / F1 for containment.

---

#### `eval_single_hyperrect.py` — Single SVD-aligned Hyper-rectangle

Fits one hyper-rectangle to the SVD-aligned pre-activation embeddings of harmful construction samples, then checks which eval samples fall inside.

```bash
python src/evaluations/eval_single_hyperrect.py \
    --model "bert-guardrail" \
    --pooling "first" \
    --dataset "your-dataset.jsonl" \
    --split "train" \
    --input-col "prompt" \
    --label-col "label"
```

| Argument | Default | Description |
|---|---|---|
| `--model` / `-m` | required | HuggingFace model ID or local path |
| `--pooling` / `-p` | required | `first` (encoder/CLS) or `last` (decoder) |
| `--dataset` / `-d` | required | HuggingFace dataset ID or path to `.jsonl` |
| `--split` / `-s` | required | Dataset split (e.g. `train`) |
| `--input-col` / `-i` | required | Column containing user input text |
| `--label-col` / `-c` | required | Binary label column (`0` = harmless, `1` = harmful) |
| `--output-col` / `-o` | `None` | Optional assistant output column |
| `--n-eval` / `-n` | `100` | Held-out samples per class |
| `--batch-size` / `-b` | `2` | Batch size for embedding extraction |
| `--max-len` / `-l` | `128` | Max tokenizer sequence length |
| `--seed` | `42` | Shuffle seed for construction/eval split |

---

#### `eval_multiple_hyperrect.py` — Multiple SVD-aligned Hyper-rectangles (HDBSCAN)

Partitions the harmful construction embeddings into clusters via HDBSCAN, fits one SVD-aligned hyper-rectangle per cluster, and classifies a point as *inside* if it falls within **any** rectangle (union membership).

```bash
python src/evaluations/eval_multiple_hyperrect.py \
    --model "gpt2-guardrail" \
    --pooling "last" \
    --dataset "your-dataset.jsonl" \
    --split "train" \
    --input-col "prompt" \
    --label-col "label" \
    --min-cluster 25
```

Accepts all arguments from the table above, plus:

| Argument | Default | Description |
|---|---|---|
| `--min-cluster` | `6` | Minimum cluster size for HDBSCAN |

> Embeddings are cached under `.embedding_cache/` after first extraction to avoid redundant forward passes across runs.

---

#### `eval_gmm.py` — Gaussian Mixture Model

Fits a GMM to the harmful construction embeddings and classifies evaluation samples by log-likelihood: a point is considered *inside* the specification if its score exceeds the 5th percentile of the construction distribution.

```bash
python src/evaluations/eval_gmm.py \
    --model "gpt2-guardrail" \
    --pooling "last" \
    --dataset "your-dataset.jsonl" \
    --split "train" \
    --input-col "prompt" \
    --label-col "label" \
    --n-components 3 \
    --cov-type full
```

Accepts all arguments from the table above, plus:

| Argument | Default | Choices | Description |
|---|---|---|---|
| `--n-components` / `-k` | `2` | any int | Number of GMM mixture components |
| `--cov-type` | `diag` | `full`, `diag` | GMM covariance structure; `diag` is recommended for high-dimensional embeddings |

> Embeddings are cached identically to `eval_multiple_hyperrect.py` — cache files are shared across scripts for the same model/dataset/split/seed combination.

---

#### Shared output format

All three scripts print the same containment table to stdout:

```
=======================================================
  HYPERRECTANGLE CONTAINMENT RESULTS
=======================================================
Category              Inside  Outside   % Inside
-------------------------------------------------------
Harmful  (label=1)        87       13      87.0%
Harmless (label=0)         4       96       4.0%
-------------------------------------------------------
Total                     91      109      45.5%
=======================================================
  Precision :   95.6%
  Recall    :   87.0%
  F1        :   91.1%
=======================================================
```

**Precision** = fraction of inside-predictions that are truly harmful. **Recall** = fraction of harmful eval samples captured inside the region. **F1** is their harmonic mean.