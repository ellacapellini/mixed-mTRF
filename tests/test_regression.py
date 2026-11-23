"""Regression tests: the refactored package must reproduce the legacy scripts' results (floats to rel 1e-6; everything else exactly).
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


def _assert_close(a, b, path="row", rel=1e-6, abs_=1e-9):
    """Same structure; ints/bools/strings exactly equal; floats equal to a tight tolerance.
    (Exact float equality is not portable: Linux and Apple-silicon BLAS differ in the last digits.)"""
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys differ"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}", rel, abs_)
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]", rel, abs_)
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=rel, abs=abs_), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


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
    _assert_close(_rows(out), _rows(FIX / f"{name}.jsonl"))


def test_summarizer_matches_legacy(tmp_path):
    ref = (FIX / "summarize_ref.txt").read_text()
    r = subprocess.run([sys.executable, "-m", "mixed_trf", "summarize", "--glob", str(FIX / "results_in" / "*.jsonl"), "--table"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ref.strip()