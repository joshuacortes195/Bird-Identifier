# Wildlife Classifier — task runner.
# On macOS/Linux/CI these targets work directly. On Windows (no make), use
# tasks.ps1 (e.g. `./tasks.ps1 lint`) or run the underlying commands shown here.

PYTHON ?= python
PKG    := src/wildlife scripts tests

.PHONY: help install install-cpu install-cuda lint fmt test test-fast smoke train eval serve demo export clean

help:
	@echo "Targets: install install-cpu install-cuda lint fmt test smoke train eval serve demo export clean"

# Editable install with dev+serve+optimize extras. Torch is installed separately
# (see install-cpu / install-cuda) because the wheel differs per platform.
install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e ".[dev,serve,optimize,tracking]"

install-cuda:
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

install-cpu:
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow and not gpu"

smoke:
	$(PYTHON) scripts/train.py train=smoke

train:
	$(PYTHON) scripts/train.py

eval:
	$(PYTHON) scripts/evaluate.py

export:
	$(PYTHON) scripts/export.py

serve:
	$(PYTHON) -m uvicorn wildlife.serve.app:app --host 0.0.0.0 --port 8000

demo:
	$(PYTHON) -m wildlife.serve.gradio_demo

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info
