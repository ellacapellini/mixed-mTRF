"""Regression tests: the refactored package must reproduce the legacy scripts' results BIT-FOR-BIT.
Fixtures were produced by the pre-refactor erp_testbed.py / ieeg_stage1.py / summarize_ieeg1.py."""
import json
import pathlib
import subprocess
import sys

import pytest

FIX = pathlib.Path(__file__).parent / "fixtures"
BASE = ["--n_patients", "4", "--electrodes", "20", "--n_times", "40000", "--select_k", "4", "--snr", "0.1",
        "--seed", "3", "--modes", "null", "gain", "latency", "width"]


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if not k.startswith("t_")}   # timings differ run to run
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return x


def _rows(path):
    return [_strip(json.loads(l)) for l in open(path) if l.strip()]


@pytest.mark.parametrize("name,extra", [
    ("stage1_firstorder", ["--band", "lfp"]),
    ("stage1_binned", ["--band", "hg", "--estimator", "binned"]),
    ("stage1_rhythm", ["--band", "lfp", "--estimator", "binned", "--rhythm", "isochronous", "--violation_p", "0.1"]),
])
def test_stage1_matches_legacy(tmp_path, name, extra):
    from mixed_trf.stage1.pipeline import main
    out = tmp_path / "o.jsonl"
    main(BASE + extra + ["--out", str(out)])
    assert _rows(out) == _rows(FIX / f"{name}.jsonl")


def test_summarizer_matches_legacy(tmp_path):
    ref = (FIX / "summarize_ref.txt").read_text()
    r = subprocess.run([sys.executable, "-m", "mixed_trf", "summarize", "--glob", str(FIX / "results_in" / "*.jsonl"), "--table"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ref.strip()
