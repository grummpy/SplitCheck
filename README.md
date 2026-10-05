# SplitCheck

![SplitCheck cover](assets/splitcheck-cover.jpg)

SplitCheck is a local machine-learning leakage lab. It finds train/test overlap — copied rows, near-copies, and shared group ids — and it measures how a leaky protocol inflates scores on a synthetic clinic-visit table.

Every table is generated locally from a NumPy seed.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the lab

One command writes the comparison. The same Markdown is printed to the terminal.

```bash
python -m splitcheck lab --seed 42 --output reports/lab-report.md
```

`reports/` is gitignored. Omit `--output` to print the report only.

On seed 42 the headline logistic regressions are:

| Protocol | ROC-AUC | Accuracy |
| --- | ---: | ---: |
| Leaky: random split, encode and scale on all rows | 1.0000 | 1.0000 |
| Clean: group split, encode and scale on train | 0.7759 | 0.6778 |

The gap is +0.2241 ROC-AUC. The clean score matches a model that sees only age, BMI, and blood pressure (ROC-AUC 0.7774). A random forest that also sees a patient fingerprint scores 0.9871 on the random split and 0.7059 when whole patients are held out. Every planted fixture in that report is `PASS`.

The full write-up of why the leaky score is inflated is the report itself. The cases are also described in [docs/leakage-cases.md](docs/leakage-cases.md).

## Reproduce from a seed

The documented seed is **42** (`splitcheck.constants.DEFAULT_SEED`).

```bash
python -m splitcheck lab --seed 42
python -m splitcheck lab --seed 42
```

Those two runs print the same report, including the dataset hash. The clinic table comes from `numpy.random.Generator` (PCG64) seeded with that integer. The row split, the patient split, and the forest all use it as `random_state`.

## What success looks like

- `pytest` exits 0.
- The report's fixture table is all `PASS`: 11 exact copies, 9 near-copies, 5 shared patient ids, the combined case, and both clean controls.
- The random experiment split shares patients across train and test. The group split shares none. Neither split has duplicate rows.
- Leaky ROC-AUC is essentially 1. Clean ROC-AUC stays with the clinical-only reference, and the gap is large.
- A second run with seed 42 matches the first.

```bash
pytest
python -m splitcheck lab --seed 42
```

## Audit a split you already have

```bash
python -m splitcheck audit \
  --train train.csv \
  --test test.csv \
  --group patient_id \
  --target y
```

`--strict` exits with status 1 when any leakage is found. Near-copies use a Chebyshev tolerance of `0.001` on the numeric features (`--near-atol`). Categorical features have to match exactly.

## Layout

```
src/splitcheck/    detectors, synthetic data, the two protocols, report, CLI
tests/             detectors, planted fixtures, leaky vs clean scores, seed match
docs/leakage-cases.md
assets/splitcheck-cover.jpg
```

`python -m splitcheck` and the `splitcheck` console script call the same entry point.

## License

MIT. See [LICENSE](LICENSE).
