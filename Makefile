.PHONY: install lint test process features train predict evaluate submit pipeline clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/ --ignore-missing-imports

test:
	pytest tests/ -v --tb=short

process:
	python -m src.data.process

features:
	python -m src.features.build

train:
	python -m src.models.train

predict:
	python -m src.models.predict

evaluate:
	python -m src.models.evaluate

submit:
	python -m src.submission.format

pipeline: process features train predict evaluate submit

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
