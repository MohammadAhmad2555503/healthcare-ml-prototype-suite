# Eye Disease Screening AI

Production-style healthcare AI prototype for Ophthalmology imaging.

## Use Case

Retinal and eye image screening support for common ophthalmic disease categories.

## What Is Included

- Data manifest/loading and validation
- Reproducible training pipeline
- Logistic Regression, Random Forest, and Support Vector Machine model comparison
- Automatic best-model selection
- Model artifacts under `models/`
- Metrics, confusion matrix, class distribution, feature importance, and comparison reports under `reports/`
- Multi-page Streamlit app
- Unit tests
- Dockerfile and docker-compose.yml

## Data

For image datasets, the generated `data/manifest.csv` references images inside `datasets.zip`. Copy `datasets.zip` to `data/datasets.zip` or set `HEALTH_DATASET_ZIP` before retraining.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --max-per-class 250
```

## Run App

```bash
streamlit run app/Home.py
```

## Test

```bash
python -m unittest discover -s tests
```

## Docker

```bash
docker build -t eyediseaseai .
docker run -p 8501:8501 eyediseaseai
```

## Clinical Safety

This is an MVP decision support prototype. It is not a standalone diagnostic device and must not be used for clinical care without external validation, clinician review, quality management, privacy/security review, monitoring, and regulatory assessment.

