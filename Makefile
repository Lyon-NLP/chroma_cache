install:
	pip install -e .

install-dev:
	pip install ".[dev]"

lint:
	ruff format .
	ruff check . --fix

lint-check:
	ruff format . --check
	ruff check **/*.py 

test:
	pytest -n auto 