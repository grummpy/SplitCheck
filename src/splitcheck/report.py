"""Render a lab result as Markdown. The text contains no timestamps, so a seed reprints it."""

from __future__ import annotations

from splitcheck.constants import NEAR_ATOL, TARGET_ENCODING_SMOOTHING
from splitcheck.detectors import SplitAudit
from splitcheck.pipelines import LabResult, Score


def render_report(result: LabResult) -> str:
    leaky = result.score("leaky")
    clean = result.score("clean")
    lines: list[str] = [
        "# SplitCheck lab report",
        "",
        f"Seed: {result.seed}",
        "",
        "This report compares one leaky train/test protocol with one correct protocol",
        "on a synthetic clinic-visit table. Both headline runs use logistic regression.",
        "The leaky score is higher because the protocol puts patient identity, and the",
        "patient's own label, into the feature matrix.",
        "",
        "## Dataset",
        "",
        f"- Rows: {result.dataset_rows}",
        f"- Patients: {result.dataset_patients}",
        f"- Visits per patient: {result.visits_per_patient}",
        f"- Positive rate: {result.positive_rate:.4f}",
        f"- Dataset sha256 (first 16 hex chars): `{result.dataset_sha256}`",
        "- Outcome `y`: 1 when a patient-level risk draw is positive. Every visit from that patient shares the label.",
        "- Clinical features `age`, `bmi`, `systolic_bp`: a weak function of the same risk, plus visit-level noise.",
        "- `billing_code`: one code per patient. It identifies the patient. It is not a lab test.",
        "- `fp0`–`fp3`: a fingerprint drawn independently of the label and held nearly constant across that patient's visits.",
        "- Group key: `patient_id`.",
        "",
        "## Headline result",
        "",
        "The leaky protocol draws a random row split, target-encodes `billing_code` with every row's label,",
        "and fits the scaler on every row. The clean protocol keeps every visit from a patient on one side",
        "of the split, fits the encoder and the scaler on the training patients, and maps an unseen code",
        "to the training base rate.",
        "",
        _metric_table([leaky, clean]),
        "",
        _gap_line(leaky, clean),
        "",
        "## Why the leaky score is inflated",
        "",
        "The clinical columns are the same in both protocols. The extra performance comes from `billing_code`.",
        f"With smoothing m = {TARGET_ENCODING_SMOOTHING:g}, the encoder maps a code to",
        "`(sum of y + m * mean of y) / (count + m)`. A code belongs to one patient and the label is",
        "constant across that patient's visits, so the encoded value is a shrunk copy of the label",
        "whenever the patient's own rows are in the fit.",
        "",
        "Each bug is enough on its own. The middle rows turn one bug on and leave the other off.",
        "The reference row drops `billing_code` and keeps the group split, which is the clinical signal",
        "the clean protocol is supposed to recover.",
        "",
        _metric_table(
            [
                result.score("leaky"),
                result.score("group_leak"),
                result.score("preprocess_leak"),
                result.score("clean"),
                result.score("clinical_only"),
            ]
        ),
        "",
        _ablation_prose(result),
        "",
        _scaler_prose(result.scaler_only_auc_delta),
        "",
        "A random forest on the clinical features plus the fingerprint shows the split bug with no target encoding.",
        "The fingerprint does not depend on the label. It is stable across a patient's visits, so a forest",
        "that has seen one visit can recognize the others. Held-out patients fall outside those leaves,",
        "and the forest has to use the clinical features.",
        "",
        _metric_table([result.score("forest_random"), result.score("forest_group")]),
        "",
        "## Leakage audit of the experiment splits",
        "",
        "Duplicate checks use the clinical features, the fingerprint, and `billing_code`.",
        f"A near-duplicate is a test row within Chebyshev distance {NEAR_ATOL:g} of a train row",
        "that also matches on `billing_code`. Sister visits share a billing code, but the vitals",
        "move between visits, so they are group leakage rather than duplicate rows.",
        "",
        _audit_table(result),
        "",
        "## Planted fixture audit",
        "",
        "These frames are built separately from the clinic table. The leaks are inserted by copying",
        "rows or by reusing patient ids. A row passes when the detector counts match the planted counts.",
        "",
        _fixture_table(result),
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python -m splitcheck lab --seed {result.seed}",
        "```",
        "",
        "The documented default seed is 42 (`splitcheck.constants.DEFAULT_SEED`).",
        "The clinic table comes from `numpy.random.Generator` (PCG64) seeded with the integer above.",
        "The row split, the group split, and the forest all take that same integer as `random_state`.",
        "A second run with the same seed reprints this report, including the dataset hash above.",
        "",
    ]
    return "\n".join(lines)


def render_audit(audit: SplitAudit, title: str = "SplitCheck audit") -> str:
    lines = [
        f"# {title}",
        "",
        f"- Train rows: {audit.n_train}",
        f"- Test rows: {audit.n_test}",
        f"- Exact duplicate test rows: {audit.exact_duplicate_rows}",
        f"- Near-duplicate test rows: {audit.near_duplicate_rows} (Chebyshev atol {audit.near_atol:g})",
        f"- Overlapping groups: {len(audit.overlapping_groups)}",
        f"- Train rows in those groups: {audit.train_rows_with_overlapping_group}",
        f"- Test rows in those groups: {audit.test_rows_with_overlapping_group}",
    ]
    if audit.overlapping_groups:
        preview = ", ".join(audit.overlapping_groups[:8])
        extra = len(audit.overlapping_groups) - 8
        if extra > 0:
            preview = f"{preview}, ... ({extra} more)"
        lines.append(f"- Group ids: {preview}")
    lines.append("")
    if audit.has_leakage:
        lines.append("Findings: leakage is present.")
    else:
        lines.append("Findings: no duplicate rows and no shared groups.")
    lines.append("")
    return "\n".join(lines)


def _metric_table(scores: list[Score]) -> str:
    header = "| Protocol | Split | Fit | Model | Accuracy | F1 | ROC-AUC | AP |"
    rule = "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"
    body = [
        "| {title} | {split} | {fit} | {model} | {accuracy:.4f} | {f1:.4f} | {roc_auc:.4f} | {average_precision:.4f} |".format(
            title=score.title,
            split=score.split,
            fit=score.fit,
            model=score.model,
            accuracy=score.accuracy,
            f1=score.f1,
            roc_auc=score.roc_auc,
            average_precision=score.average_precision,
        )
        for score in scores
    ]
    return "\n".join([header, rule, *body])


def _gap_line(leaky: Score, clean: Score) -> str:
    return (
        f"Gap (leaky minus clean): accuracy {leaky.accuracy - clean.accuracy:+.4f}, "
        f"F1 {leaky.f1 - clean.f1:+.4f}, "
        f"ROC-AUC {leaky.roc_auc - clean.roc_auc:+.4f}, "
        f"average precision {leaky.average_precision - clean.average_precision:+.4f}."
    )


def _ablation_prose(result: LabResult) -> str:
    group_leak = result.score("group_leak")
    preprocess = result.score("preprocess_leak")
    clean = result.score("clean")
    clinical = result.score("clinical_only")
    return "\n".join(
        [
            f"Random split, encoder fit on train only, reaches ROC-AUC {group_leak.roc_auc:.4f}. "
            "The encoder never sees the test rows, but sister visits of the same patient are in the training rows. "
            "Those visits carry the label, they share `billing_code`, and the test visits receive that code's mean.",
            "",
            f"Group split, encoder fit on all rows, reaches ROC-AUC {preprocess.roc_auc:.4f}. "
            "No patient is shared. The encoder still saw the held-out patient's label before the split, "
            "so the code mean on the test rows is that patient's label.",
            "",
            f"The clean protocol reaches ROC-AUC {clean.roc_auc:.4f}. "
            f"A logistic regression on the clinical features alone, with the same group split, "
            f"reaches ROC-AUC {clinical.roc_auc:.4f}. "
            "Test codes are unseen, the encoded column is the training base rate, and the score is the clinical association.",
        ]
    )


def _scaler_prose(delta: float) -> str:
    return (
        "The leaky protocol also fits `StandardScaler` on every row. "
        "That leaks test-set means and variances. "
        f"With a group split and no target encoding, fitting the scaler on all rows instead of the training rows "
        f"changes ROC-AUC by {delta:+.6f}. "
        "The inflated headline score comes from the identifier and the split."
    )


def _audit_table(result: LabResult) -> str:
    header = "| Split | Test rows | Overlapping patients | Test rows in those patients | Exact duplicate rows | Near-duplicate rows |"
    rule = "| --- | ---: | ---: | ---: | ---: | ---: |"
    return "\n".join(
        [
            header,
            rule,
            _audit_row("Random", result.random_audit),
            _audit_row("Group", result.group_audit),
        ]
    )


def _audit_row(name: str, audit: SplitAudit) -> str:
    return (
        f"| {name} | {audit.n_test} | {len(audit.overlapping_groups)} | {audit.test_rows_with_overlapping_group} "
        f"| {audit.exact_duplicate_rows} | {audit.near_duplicate_rows} |"
    )


def _fixture_table(result: LabResult) -> str:
    header = "| Case | Exact | Near | Groups | Expected | Result |"
    rule = "| --- | ---: | ---: | ---: | --- | --- |"
    rows = [header, rule]
    for spec in result.fixtures:
        expected = f"{spec.expect_exact} exact, {spec.expect_near} near, {spec.expect_groups} groups"
        status = "PASS" if spec.passed else "FAIL"
        rows.append(
            f"| `{spec.name}` | {spec.exact_duplicate_rows} | {spec.near_duplicate_rows} "
            f"| {spec.overlapping_groups} | {expected} | {status} |"
        )
    return "\n".join(rows)
