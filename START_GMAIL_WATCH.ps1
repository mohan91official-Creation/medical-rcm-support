$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment Python was not found. Create .venv and install requirements first."
}

& $python "gmail_workflow.py" --watch --send --interval 60
