install:
	pip install -e .

install-dev:
	pip install ".[dev]"

lint:
	black chromacache/
	ruff format .
	ruff check . --fix

lint-check:
	ruff format . --check
	ruff check .

test:
	pytest tests/