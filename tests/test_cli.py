"""The lab command is the one-shot report. The audit command reads local CSVs."""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from splitcheck.cli import main
from splitcheck.constants import DEFAULT_SEED

ROOT = Path(__file__).resolve().parents[1]


def test_lab_command_prints_the_report(tmp_path, capsys):
    output = tmp_path / "lab-report.md"
    status = main(["lab", "--seed", str(DEFAULT_SEED), "--output", str(output)])
    captured = capsys.readouterr()
    assert status == 0
    assert f"Seed: {DEFAULT_SEED}" in captured.out
    assert "Why the leaky score is inflated" in captured.out
    assert "FAIL" not in captured.out
    assert output.read_text(encoding="utf-8") == captured.out


def test_module_entry_point_runs():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-m", "splitcheck", "lab", "--seed", "42"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert completed.returncode == 0
    assert "Seed: 42" in completed.stdout
    assert "Why the leaky score is inflated" in completed.stdout


def test_audit_strict_exits_on_shared_groups_and_passes_a_clean_pair(tmp_path, capsys):
    train = tmp_path / "train.csv"
    leaked = tmp_path / "leaked.csv"
    clean = tmp_path / "clean.csv"
    pd.DataFrame({"x0": [0.0, 1.0], "patient_id": ["A", "B"], "y": [0, 1]}).to_csv(train, index=False)
    pd.DataFrame({"x0": [8.0, 9.0], "patient_id": ["B", "C"], "y": [1, 0]}).to_csv(leaked, index=False)
    pd.DataFrame({"x0": [8.0, 9.0], "patient_id": ["C", "D"], "y": [1, 0]}).to_csv(clean, index=False)

    leaked_status = main(
        ["audit", "--train", str(train), "--test", str(leaked), "--group", "patient_id", "--target", "y", "--strict"]
    )
    leaked_out = capsys.readouterr().out
    assert leaked_status == 1
    assert "Overlapping groups: 1" in leaked_out

    clean_status = main(
        ["audit", "--train", str(train), "--test", str(clean), "--group", "patient_id", "--target", "y", "--strict"]
    )
    clean_out = capsys.readouterr().out
    assert clean_status == 0
    assert "no duplicate rows and no shared groups" in clean_out


def test_audit_rejects_a_missing_file(tmp_path, capsys):
    status = main(["audit", "--train", str(tmp_path / "missing.csv"), "--test", str(tmp_path / "also.csv")])
    assert status == 1
    assert "splitcheck:" in capsys.readouterr().err


def test_audit_rejects_default_schema_loss_and_allows_an_explicit_subset(tmp_path, capsys):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame({"x": [1.0], "z": [2.0]}).to_csv(train, index=False)
    pd.DataFrame({"x": [1.0]}).to_csv(test, index=False)

    assert main(["audit", "--train", str(train), "--test", str(test)]) == 1
    assert "missing from test: z" in capsys.readouterr().err

    assert main(["audit", "--train", str(train), "--test", str(test), "--features", "x"]) == 0
