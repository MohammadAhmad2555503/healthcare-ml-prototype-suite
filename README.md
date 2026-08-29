# Healthcare ML Prototype Suite

A portfolio-ready suite of healthcare machine-learning prototypes covering structured clinical prediction, imaging triage, screening workflows, reporting, dashboards, Docker setup, and tests.

This repository is designed for employer review. It shows breadth across clinical domains while keeping each project inspectable as a standalone engineering artifact.

## What This Demonstrates

- End-to-end Python project structure across multiple healthcare use cases.
- Streamlit dashboards for interactive review and workflow exploration.
- Reproducible training and prediction paths with saved model metadata.
- Small sample datasets or manifests where practical, with larger runtime artifacts excluded.
- Docker and dependency files for easier local review.
- Tests, reports, and documentation organized close to each project.
- Responsible framing for healthcare prototypes, including limitations and non-clinical scope.

## Repository Structure

```text
healthcare-ml-prototype-suite/
  README.md
  PROJECTS.md
  EMPLOYER_SHOWCASE.md
  docs/
    ARCHITECTURE.md
    REVIEW_GUIDE.md
    SAFETY_AND_LIMITATIONS.md
    RESUME_BULLETS.md
    PORTFOLIO_CARD.md
  projects/
    breast-cancer-ai/
    brain-tumour-ai/
    eye-disease-ai/
    heart-disease-ai/
    lung-cancer-ai/
    brain-tumor-detection-system/
    breast-cancer-decision-support/
    eye-disease-screening-system/
    heart-disease-risk-assessment/
    lung-cancer-imaging-system/
```

## Included Projects

| Project | Category | Focus |
| --- | --- | --- |
| Breast Cancer AI | Structured Clinical Prediction | Malignancy prediction, model comparison, explainability, dashboard pages, and tests |
| Brain Tumour AI | Medical Imaging | Neuro-oncology image triage, manifests, model metadata, dashboard pages, and reports |
| Eye Disease AI | Medical Imaging | Ophthalmology screening workflow with image manifests, training path, dashboard pages, and tests |
| Heart Disease AI | Structured Clinical Prediction | Cardiovascular risk prediction using tabular data, reports, and dashboard pages |
| Lung Cancer AI | Medical Imaging | Thoracic imaging triage workflow with manifests, model comparison, and Streamlit pages |
| Brain Tumor Detection System | Clinical Workflow System | Detection-oriented triage system with reporting, service code, dashboard pages, and tests |
| Breast Cancer Decision Support | Clinical Workflow System | Decision-support workflow for risk review, reporting, dashboard pages, and tests |
| Eye Disease Screening System | Clinical Workflow System | Screening system for eye disease categories with reports, dashboard pages, and tests |
| Heart Disease Risk Assessment | Clinical Workflow System | Risk-assessment workflow with prediction services, reports, dashboard pages, and tests |
| Lung Cancer Imaging System | Clinical Workflow System | Imaging review workflow with model metadata, reporting, dashboard pages, and tests |

## How To Review

Start with [PROJECTS.md](PROJECTS.md) for a full project matrix, then open any project under `projects/`.

For a quick employer review, inspect:

1. `projects/heart-disease-risk-assessment/`
2. `projects/breast-cancer-decision-support/`
3. `projects/brain-tumor-detection-system/`
4. `projects/eye-disease-screening-system/`
5. `projects/lung-cancer-imaging-system/`

Each project keeps its own README, source folders, tests, reports, Docker files, and configuration where available.

## Running A Project

Open one project folder and follow its README. Most projects follow this pattern:

```powershell
cd projects/<project-folder>
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/Home.py
```

Some projects use `environment.yml` instead of `requirements.txt`. In that case, follow the setup notes in that project README.

## Healthcare Scope

These are educational portfolio prototypes for software and data engineering review. They are not clinical products, medical devices, diagnostic tools, or replacements for professional medical judgment.

See [docs/SAFETY_AND_LIMITATIONS.md](docs/SAFETY_AND_LIMITATIONS.md) for the full scope statement.

## GitHub Readiness

This suite was assembled to be upload-friendly:

- No nested Git repositories inside `projects/`.
- No local virtual environments, dependency folders, logs, PID files, database files, or archive files.
- README and employer-review documents are included at the suite level.
- Individual project READMEs remain available for deeper inspection.
- Small project assets are included where useful for review.
