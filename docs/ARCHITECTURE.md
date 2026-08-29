# Architecture

The suite is organized as a portfolio-level repository with each prototype preserved as a self-contained project under `projects/`.

## Suite Layer

The root files explain the whole body of work:

- `README.md` gives the main employer-facing overview.
- `PROJECTS.md` maps every included prototype.
- `EMPLOYER_SHOWCASE.md` highlights what to review first.
- `docs/` contains architecture, safety, review, resume, and portfolio notes.

## Project Layer

Most projects follow a similar structure:

```text
project/
  app/       Streamlit pages and interaction layer
  src/       reusable training, prediction, validation, and reporting code
  data/      small sample datasets or manifests
  models/    saved model metadata and compact model assets where practical
  reports/   evaluation notes, metrics, and review material
  tests/     focused tests for project behavior
  docs/      additional project-level notes
```

## Common Workflow

1. Data assets or manifests define the review inputs.
2. Training or loading code prepares the model layer.
3. Prediction services expose reusable scoring behavior.
4. Streamlit pages provide an interactive review surface.
5. Reports and tests make the project easier to inspect and verify.

## Design Priorities

- Keep every prototype understandable without needing the rest of the suite.
- Prefer small, reviewable assets over large runtime artifacts.
- Keep documentation close to the code it describes.
- Make healthcare limitations visible and direct.
- Make the repository useful for both quick employer screening and deeper technical review.
