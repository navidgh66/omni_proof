.PHONY: install test lint format build clean publish-test publish

install:
	pip install -e ".[dev]"

test:
	pytest -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

build:
	pip install build && python -m build

clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info

publish-test:
	pip install twine && twine upload --repository testpypi dist/*

publish:
	pip install twine && twine upload dist/*
