# Planted leakage cases

SplitCheck builds every table locally from a NumPy Generator. Nothing is downloaded at runtime.

The documented seed is **42** (`splitcheck.constants.DEFAULT_SEED`). The clinic experiment uses that integer directly. The hand-built fixtures draw from child streams of the same integer, so one seed fixes both.

## What the detector calls leakage

`audit_split` looks at three things.

**Exact duplicate.** A test row matches a train row on every categorical feature, and the Chebyshev distance on the numeric features is at most `1e-8`.

**Near-duplicate.** Same categorical match, and the Chebyshev distance is greater than `1e-8` and at most `1e-3`. Planted near-copies add `5e-4` to each numeric feature, which lands in that band. Rows that merely share a category are not near-duplicates.

**Group overlap.** The same group id (compared as a string) occurs in both frames. The rows do not have to look alike. For the clinic table the group is `patient_id`.

A clean split has all three counts at zero.

## Hand-built fixtures

These frames are not the clinic experiment. The leaks are inserted by copying rows or by reusing ids, and the expected counts are constants (`PLANTED_EXACT_COPIES = 11`, `PLANTED_NEAR_COPIES = 9`, five ids `SHARE0`–`SHARE4`).

| Case | What was planted | What must be found |
| --- | --- | --- |
| `exact_row_copies` | 11 test rows are feature-for-feature copies of train rows. Their patient ids are new, so this is not group leakage. | 11 exact, 0 near, 0 groups |
| `near_row_copies` | 9 test rows are those copies plus `5e-4` on `x0`, `x1`, and `x2`. The categorical `site` is unchanged. | 0 exact, 9 near, 0 groups |
| `shared_patient_ids` | Five patient ids have visits on both sides. The test visits sit far from the train visits in feature space. | 0 exact, 0 near, those five ids |
| `all_three_together` | All of the above in one train/test pair. | 11 exact, 9 near, those five ids |
| `clean_separated_rows` | Independent rows. Test features are centered far from train. Disjoint ids. | all zeros |
| `clean_repeated_measures` | Several visits per patient, and each patient is kept on one side. Visit noise is large enough that visits are not near-copies. | all zeros |

`all_three_together` exists so a detector that stops at the first kind of leak still has to count the other two.

## The clinic experiment

`make_clinic_dataset` builds 360 patients with 4 visits each (1440 rows).

- A patient-level risk draw sets `y`. The label does not change between visits.
- `age`, `bmi`, and `systolic_bp` depend weakly on that risk and then move from visit to visit.
- `billing_code` is one code per patient. It is an identifier.
- `fp0`–`fp3` are a fingerprint: independent of the label, with visit noise of `0.01`.

Sister visits are not near-duplicates. Blood pressure and age move by more than the `1e-3` tolerance. The random split is still leakage, because the patient id is shared. The audit is supposed to say so, and to stay quiet on the group split.

### Leaky protocol

1. Target-encode `billing_code` with **every** row's label. Smoothing is `m = 1`: `(sum of y + m * mean of y) / (count + m)`.
2. Fit `StandardScaler` on **every** row.
3. Split rows at random (`train_test_split`, `test_size=0.25`, stratified on `y`).
4. Fit logistic regression on the training rows and score the test rows.

Because a code belongs to one patient, the encoded column is a shrunk copy of that patient's label whenever the patient's rows were in the fit.

### Clean protocol

1. `GroupShuffleSplit` on `patient_id` (`test_size=0.25` of **patients**, `random_state` = the seed).
2. Fit the encoder on the training patients only. Held-out codes become the training base rate.
3. Fit the scaler on the training matrix only.
4. Fit the same logistic regression and score the held-out patients.

### Why each bug is enough

The lab also scores the two bugs separately, on the same model and the same raw columns.

| Protocol | Split | Where the encoder and scaler are fit | What leaks |
| --- | --- | --- | --- |
| Leaky | random rows | all rows | both bugs |
| Group leak only | random rows | training rows | the patient is on both sides, so the train-only encoding still sees the label |
| Preprocess leak only | by patient | all rows | no shared patient, but the encoder already saw the test label |
| Clean | by patient | training rows | neither |

A further reference row drops `billing_code` and uses the group split. Its ROC-AUC is the clinical association. The clean protocol should land next to it.

Fitting the scaler on all rows is a real leak of test-set moments. On this table, with a group split and no encoding, it barely moves logistic regression. The report prints that delta so the scaler is not credited with the gap.

### Fingerprints, without target encoding

A random forest sees `age`, `bmi`, `systolic_bp`, and `fp0`–`fp3`. The scaler is fit on the training rows in both runs. The only difference is the split.

The fingerprint does not cause the label. It is stable across a patient's visits, so a forest can memorize a patient it has already seen. On a group split those patients are absent, and the forest has to use the clinical features. That gap is group leakage with no target encoding at all.

## What success looks like

- Every fixture row in the report is `PASS`.
- The random experiment split lists overlapping patients, and zero duplicate rows.
- The group experiment split lists no overlapping patients and zero duplicate rows.
- Leaky ROC-AUC is essentially perfect. Clean ROC-AUC sits with the clinical-only reference, well below the leaky score.
- Running `python -m splitcheck lab --seed 42` a second time prints the same report, including the dataset hash.

## What this lab does not claim

It does not detect a feature that was built from future information inside a single row, unless that feature makes the row a duplicate or a group collision. It does not treat a small scaler-moment leak as the thing that produced a perfect score. The report says when the scaler delta is negligible.
