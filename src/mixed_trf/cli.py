"""Command line:  python -m mixed_trf <command> [args]

  models                    list the models of the ladder
  selftest                  likelihood identity + tiny end-to-end Bayesian fit
  stage1  [...]             iEEG-like simulation + fast ridge/binned estimator (ieeg_stage1 arguments)
  testbed [...]             single-electrode-family ERP testbed (erp_testbed arguments)
  stage2  --model M [...]   fit ONE Bayesian model on simulated data (one model at a time)
  summarize [...]           gates/verdict table over stage-1 jsonl results
  plot-learning [...]       learning-evolution figures from stage-2 jsonl results
"""
from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "models":
        from .models.registry import MODELS
        for k, v in MODELS.items():
            print(f"{k:14s} {v}")
    elif cmd == "selftest":
        from .models.pop_trf import _selftest; _selftest()
    elif cmd == "stage1":
        from .stage1.pipeline import main as m; m(rest)
    elif cmd == "testbed":
        from .simulation.testbed import main as m; m(rest)
    elif cmd == "stage2":
        from .experiments.stage2 import main as m; m(rest)
    elif cmd == "summarize":
        import runpy
        sys.argv = ["summarize"] + rest
        runpy.run_module("mixed_trf.evaluation.summarize", run_name="__main__")
    elif cmd == "plot-learning":
        from .evaluation.learning import main as m; m(rest)
    else:
        print(f"unknown command {cmd!r}\n{__doc__}"); return 2
    return 0
