# Ticket Priority ML Service

This repository is an end-to-end applied machine learning project for support-ticket triage. It predicts both the operational `queue` and the business `priority` from ticket text, logs experiments in MLflow, and serves fixed promoted demo models through a FastAPI API, a Streamlit UI, and Docker packaging.

## What This Repository Demonstrates

- bilingual ticket preprocessing for English and German support data
- shared cross-validation for two related classification tasks
- classical ML experimentation and model comparison with MLflow tracking
- fixed promoted demo assets under [`serving_assets/`](serving_assets/) for a reproducible portfolio demo
- FastAPI serving, Streamlit UI, and Docker packaging around the trained models

## Quick Start

For the fastest local demo path:

1. Clone the repository with Git LFS enabled.
2. Create and activate a virtual environment.
3. Install `requirements-app.txt`.
4. Download the NLTK stopword corpus.
5. Start FastAPI, then start Streamlit.
6. Open `http://127.0.0.1:8501` for the UI or `http://127.0.0.1:8000/docs` for the API docs.

## Problem Statement / Use Case

Support teams often need to decide two things as soon as a ticket arrives:

1. Which operational queue should handle it
2. How urgent it is

This project automates both decisions from free-text ticket content so routing and prioritization can happen more consistently and quickly.

## Dataset And License Scope

- Default training data path: `data/aa_dataset-tickets-multi-lang-5-2-50-version.csv`
- Upstream source: Kaggle dataset *Multilingual Customer Support Tickets* by Tobias Bueck
- Upstream dataset license: `CC BY-NC 4.0`
- Repository source code license: MIT, see [LICENSE](LICENSE)

The model uses ticket `subject` and `body`, plus optional `language` metadata during preprocessing and serving. It predicts 10 queue classes and 3 priority labels (`low`, `medium`, `high`).

The checked-in assets under [`serving_assets/`](serving_assets/) are kept as demo artifacts so the repository stays runnable as a portfolio project. Reuse of the dataset or derived artifacts should be reviewed against the upstream dataset terms; they are not relicensed under the repository's MIT license.

## Modeling Approach

### Preprocessing

The preprocessing pipeline in [`src/preprocessing.py`](src/preprocessing.py):

- concatenates `subject` and `body`
- normalizes Unicode with `NFKC`
- lowercases text
- replaces URLs, email addresses, and numbers with placeholders
- removes punctuation and collapses whitespace
- applies English/German stop-word removal when NLTK stopwords are available

At inference time, the API accepts optional `language` metadata (`en` / `de`). If it is omitted, the service falls back to a generic path without language-specific stop-word filtering.

### Features And Models

The promoted model uses:

- word-level TF-IDF
- `ngram_range=(1, 3)`
- `min_df=1`
- `max_df=0.95`
- task-specific `LinearSVC` classifiers
- `class_weight="balanced"`
- `max_iter=5000`

Promoted hyperparameters:

- queue: `C=16`
- priority: `C=12`

These models were selected after comparing a Logistic Regression baseline, wider n-gram variants, and alternative preprocessing choices. The final `LinearSVC` setup gave the best quality/complexity trade-off among the experiments that materially changed the system.

### Training And Serving

Training runs are logged to MLflow from [`train.py`](train.py), including metrics, per-class summaries, confusion summaries, run configuration JSON, and serialized model artifacts.

The published demo does not serve "latest run wins" artifacts. FastAPI and Streamlit load the fixed checked-in assets under [`serving_assets/`](serving_assets/), while training produces candidate MLflow artifacts for comparison and review.

## Headline Results

The deployed model pair is the promoted `linearsvc_1to3_Cq16p12` configuration referenced by the serving assets in [`serving_assets/`](serving_assets/).

| Task | Macro F1 (mean +/- std) | Accuracy (mean +/- std) |
| --- | ---: | ---: |
| Queue | 0.6854 +/- 0.0041 | 0.6892 +/- 0.0029 |
| Priority | 0.7108 +/- 0.0081 | 0.7204 +/- 0.0074 |

Language-specific performance is noticeably better on English tickets than on German tickets:

- Queue macro F1: English `0.7841`, German `0.5341`
- Priority macro F1: English `0.7951`, German `0.5960`

In this project, that gap appears large enough to reflect model and representation limits rather than simple fold noise.

## System Overview / Architecture

Inference flow: ticket request -> text preprocessing -> TF-IDF vectorization -> queue `LinearSVC` + priority `LinearSVC` -> JSON response with labels, runner-up labels, and margin gaps.

Serving design:

- one fixed `queue` model
- one fixed `priority` model
- FastAPI loads both models once at startup
- Streamlit calls the API rather than loading models directly

## Setup And Installation

### 1. Clone With Git LFS

The serving models are versioned with Git LFS.

```powershell
git lfs install
git clone git@github.com:feboe/ticket-priority-ml-service.git
cd ticket-priority-ml-service
git lfs pull
```

### 2. Create A Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

For the full training / development environment:

```powershell
pip install -r requirements.txt
```

For a lighter serving / demo environment:

```powershell
pip install -r requirements-app.txt
```

### 4. Download NLTK Stopwords

```powershell
.\.venv\Scripts\python -m nltk.downloader stopwords
```

The app fails gracefully if the corpus is missing, but the download is still required to enable English/German stop-word filtering.

## Testing

With the complete development environment from [`requirements.txt`](requirements.txt), run:

```powershell
python -m unittest discover -s tests -v
```

GitHub Actions uses the same full test command and installs the full development dependencies, including `mlflow`, so the public CI run exercises the training and tracking smoke path as well.

## Run Locally

The demo application serves the fixed promoted models from [`serving_assets/`](serving_assets/).

Start the FastAPI service:

```powershell
.\.venv\Scripts\uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, start Streamlit:

```powershell
$env:API_BASE_URL='http://127.0.0.1:8000'
.\.venv\Scripts\streamlit run app/ui.py
```

Endpoints:

- Streamlit UI: `http://127.0.0.1:8501`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### Training

Run a full training pass to create MLflow-tracked candidate models and metrics:

```powershell
.\.venv\Scripts\python train.py --algorithm linear_svc --run-group algo-benchmark-v1
.\.venv\Scripts\python -m mlflow ui --backend-store-uri .\mlruns
```

Running `train.py` does not update the demo automatically. Promoting selected MLflow artifacts into [`serving_assets/`](serving_assets/) is a maintainer-only release step handled separately.

## Run With Docker

Build and start the single-container MVP:

```powershell
docker build -t ticket-triage-demo .
docker run --rm -p 8000:8000 -p 8501:8501 ticket-triage-demo
```

The container installs `requirements-app.txt`, downloads the NLTK stopword corpus, starts FastAPI on port `8000`, and starts Streamlit on port `8501`.

## API Usage

### Request

`POST /predict`

```json
{
  "subject": "Critical security incident affecting the account portal",
  "body": "Multiple users report that the account portal is unavailable after suspicious login activity. We need urgent investigation, containment guidance, and an update on service restoration steps.",
  "language": "en"
}
```

### Response

The response contains:

- the original input payload
- a `queue` prediction and a `priority` prediction
- `label`, `runner_up_label`, and `margin_gap` for each task
- model metadata for the fixed promoted demo assets

## Limitations

- English performance is substantially better than German performance.
- The system relies on TF-IDF features and linear classifiers, so it has limited semantic understanding compared with transformer-based approaches.
- Queue labels with overlapping business meaning still produce systematic confusions.
- Ticket labels may contain some unavoidable ambiguity or noise where support categories are semantically close.
