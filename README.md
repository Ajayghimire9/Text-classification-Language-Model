# TextBench

Reproducible text-classification benchmarks.

TextBench compares word and character TF-IDF classifiers on a labelled CSV. The command-line workflow makes the original notebook experiments repeatable without a GPU or an external model service.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -r requirements-portfolio.txt
python -m textbench.pipeline --data examples/texts.csv
```

## Design decisions

Normalized duplicate text is removed before splitting; conflicting labels for the same text are rejected.

Candidate selection uses stratified cross-validation within the training partition. The final test partition is reserved for the selected model.

A single fitted pipeline contains vocabulary and classifier. Reports include per-class metrics, a confusion matrix, dataset hash and model checksum.

## Technology

Python, pandas, scikit-learn, TF-IDF, joblib, DVC-compatible pipeline, GitHub Actions.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

The included support-request examples are hand-written smoke-test fixtures, not a representative evaluation dataset. The maintained CLI uses linear classifiers; BERT experiments remain in the original notebook. Confidence scores are uncalibrated and the review threshold is configurable.
