#!/usr/bin/env python
"""Backward-compatible entry point: same arguments as the old ieeg_stage1.py. Equivalent to  python -m mixed_trf stage1 ..."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mixed_trf.cli import main
sys.exit(main(["stage1"] + sys.argv[1:]))
