#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cat <<'EON'

Done.
Next steps:
1) source .venv/bin/activate
2) cp .env.example .env  # and edit as needed
3) uvicorn api.main:app --host 0.0.0.0 --port 8000

EON

