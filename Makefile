.PHONY: help install install-eval install-dev test lint format figures repro clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Install runtime deps + the forminv package (uv)
	uv sync

install-eval: ## Install + provider SDKs (OpenAI/Anthropic/DeepSeek/Gemini) for live evals
	uv sync --extra eval

install-dev: ## Install + dev tooling (pytest, ruff, mypy)
	uv sync --all-groups

test: ## Run the offline test suite
	uv run pytest -q

lint: ## Lint with ruff
	uv run ruff check .

format: ## Auto-format with ruff
	uv run ruff format .

figures: ## Regenerate all paper figures from cached artifacts (no API)
	uv run --extra viz python scripts/generate_figures.py

repro: ## Recompute the headline results from cache (no API)
	uv run python scripts/scr_decomposition.py
	uv run python scripts/cir_rigorous.py
	uv run python scripts/cir_rigorous.py --data data/generated/forminv_v3_103.jsonl --flagged none --out artifacts/cir_rigorous_103.json
	uv run python scripts/leave_one_out_audit.py

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
