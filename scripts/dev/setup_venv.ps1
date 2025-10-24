param(
  [switch]$Force
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path .venv) -or $Force) {
  python -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host @"
Done.
Next steps:
1) .\\.venv\\Scripts\\Activate.ps1
2) Copy .env.example to .env and update values
3) uvicorn api.main:app --host 0.0.0.0 --port 8000
"@

