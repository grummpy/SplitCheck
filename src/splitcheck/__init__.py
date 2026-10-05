"""SplitCheck: a local lab for train/test leakage."""

from splitcheck.constants import DEFAULT_SEED
from splitcheck.detectors import SplitAudit, audit_split
from splitcheck.pipelines import LabResult, run_lab
from splitcheck.report import render_report
from splitcheck.synthetic import build_fixtures, make_clinic_dataset

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SEED",
    "LabResult",
    "SplitAudit",
    "__version__",
    "audit_split",
    "build_fixtures",
    "make_clinic_dataset",
    "render_report",
    "run_lab",
]
