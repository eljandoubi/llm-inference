.PHONY: sync lint format typecheck test check

sync:
	uv sync

lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

format:
	uv run --no-sync ruff format .
	uv run --no-sync ruff check --fix .

typecheck:
	uv run --no-sync mypy

test:
	uv run --no-sync pytest --cov

# what CI runs
check: lint typecheck test
