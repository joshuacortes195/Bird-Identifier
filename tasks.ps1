<#
Windows task runner mirroring the Makefile (make isn't installed here).
Usage:  ./tasks.ps1 <target>
Targets: install install-cpu install-cuda lint fmt test test-fast smoke train eval serve serve-dev demo export clean
#>
param(
    [Parameter(Position = 0)]
    [string]$Target = "help"
)

$ErrorActionPreference = "Stop"
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }

switch ($Target) {
    "help" {
        Write-Host "Targets: install install-cuda install-cpu lint fmt test test-fast smoke train eval serve demo export clean"
    }
    "install" {
        & $Py -m pip install -U pip
        & $Py -m pip install -e ".[dev,serve,optimize,tracking,demo]"
    }
    "install-cuda" { & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 }
    "install-cpu"  { & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu }
    "lint" {
        & $Py -m ruff check .
        & $Py -m ruff format --check .
    }
    "fmt" {
        & $Py -m ruff check --fix .
        & $Py -m ruff format .
    }
    "test"      { & $Py -m pytest }
    "test-fast" { & $Py -m pytest -m "not slow and not gpu" }
    "smoke"     { & $Py scripts/train.py train=smoke }
    "train"     { & $Py scripts/train.py }
    "eval"      { & $Py scripts/evaluate.py }
    "export"    { & $Py scripts/export.py }
    "serve"     { & $Py -m uvicorn wildlife.serve.app:app --host 0.0.0.0 --port 8000 }
    "serve-dev" {
        $env:WILDLIFE_ALLOW_STUB = "1"
        $env:WILDLIFE_TAXONOMY = "configs/taxonomy/birds.yaml"
        & $Py -m uvicorn wildlife.serve.app:app --host 0.0.0.0 --port 8000 --reload
    }
    "demo"      { & $Py -m wildlife.serve.gradio_demo }
    "clean" {
        Get-ChildItem -Recurse -Directory -Include __pycache__, .pytest_cache, .ruff_cache, *.egg-info -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    default { Write-Error "Unknown target: $Target" }
}
