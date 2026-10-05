"""Command line for the lab and for auditing a pair of CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from splitcheck import __version__
from splitcheck.constants import DEFAULT_SEED, EXACT_ATOL, NEAR_ATOL
from splitcheck.detectors import audit_split
from splitcheck.pipelines import run_lab
from splitcheck.report import render_audit, render_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="splitcheck",
        description="Detect train/test leakage and compare a leaky protocol with a correct one.",
    )
    parser.add_argument("--version", action="version", version=f"splitcheck {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lab = subparsers.add_parser("lab", help="Run the seeded leakage lab and print a Markdown report.")
    lab.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED}).")
    lab.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Also write the report to this path. The same text is printed to stdout.",
    )
    lab.set_defaults(handler=_run_lab)

    audit = subparsers.add_parser("audit", help="Audit a train CSV and a test CSV for leakage.")
    audit.add_argument("--train", required=True, type=Path, help="Training CSV.")
    audit.add_argument("--test", required=True, type=Path, help="Test CSV.")
    audit.add_argument("--group", default=None, help="Column that should not appear in both splits.")
    audit.add_argument("--target", default=None, help="Label column. Excluded from the duplicate check.")
    audit.add_argument(
        "--features",
        default=None,
        help="Comma-separated feature columns. Default: every column except --group and --target.",
    )
    audit.add_argument("--near-atol", type=float, default=NEAR_ATOL, help=f"Chebyshev near-duplicate tolerance (default: {NEAR_ATOL:g}).")
    audit.add_argument("--exact-atol", type=float, default=EXACT_ATOL, help=f"Chebyshev exact-duplicate tolerance (default: {EXACT_ATOL:g}).")
    audit.add_argument("--output", "-o", type=Path, help="Also write the audit to this path.")
    audit.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 when any leakage is found.",
    )
    audit.set_defaults(handler=_run_audit)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"splitcheck: {exc}", file=sys.stderr)
        return 1


def _run_lab(args: argparse.Namespace) -> int:
    report = render_report(run_lab(args.seed))
    _emit(report, args.output)
    return 0


def _run_audit(args: argparse.Namespace) -> int:
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    features = _feature_columns(train, test, args.features, args.group, args.target)
    audit = audit_split(
        train,
        test,
        feature_columns=features,
        group_column=args.group,
        near_atol=args.near_atol,
        exact_atol=args.exact_atol,
    )
    text = render_audit(audit)
    _emit(text, args.output)
    if args.strict and audit.has_leakage:
        return 1
    return 0


def _feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: str | None,
    group: str | None,
    target: str | None,
) -> list[str]:
    if features:
        names = [part.strip() for part in features.split(",") if part.strip()]
        if not names:
            raise ValueError("--features did not contain any column names")
        return names
    excluded = {name for name in (group, target) if name}
    names = [column for column in train.columns if column in test.columns and column not in excluded]
    if not names:
        raise ValueError("no feature columns left after excluding the group and target")
    return names


def _emit(text: str, output: Path | None) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
