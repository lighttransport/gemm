#!/usr/bin/env python3
"""Compare CUDA dumps with the oracle dumps: thin wrapper over ref/praxis/compare.py (same rel-L2 <= 0.02 / cos >= 0.9999 rule).
  ~/.venv/bin/python -I ref/cosmos3/compare.py --ref-dir tmp/cosmos3_ref --ours-dir tmp/cosmos3_out [--skip-missing] [--tol 0.02]
Only files present in --ours-dir are compared with --skip-missing (the default here)."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
if "--skip-missing" not in args: args.append("--skip-missing")
sys.exit(subprocess.call([sys.executable, "-I", os.path.join(here, "..", "praxis", "compare.py")] + args))
