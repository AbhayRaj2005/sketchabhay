#!/usr/bin/env bash
# One-shot setup:  ./setup.sh          (CPU pipeline + web UI)
#                  ./setup.sh gpu      (adds Stable Diffusion + ControlNet + rembg)
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "${1:-}" = "gpu" ]; then pip install -r requirements-gpu.txt; else pip install -r requirements.txt; fi
python scripts/check_env.py
python tests/test_pipeline.py
echo; echo "Ready.  source .venv/bin/activate && python app.py"
