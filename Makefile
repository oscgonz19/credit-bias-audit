.PHONY: setup data run test clean lint format help

CONDA_ENV = credit-bias-audit
PYTHON = python

help:
	@echo "Credit Bias Audit - Available targets:"
	@echo ""
	@echo "  setup    - Create conda environment from environment.yml"
	@echo "  data     - Download and prepare datasets"
	@echo "  run      - Run full audit pipeline"
	@echo "  test     - Run smoke tests with small dataset"
	@echo "  lint     - Check code style"
	@echo "  format   - Format code with black and isort"
	@echo "  clean    - Remove generated files"
	@echo ""
	@echo "Examples:"
	@echo "  make setup              # Create conda environment"
	@echo "  make data               # Download German Credit dataset"
	@echo "  make run                # Run audit with defaults"
	@echo "  make run ARGS='--model rf --protected-attr age'"

# Environment setup
setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	@echo ""
	@echo "Environment created. Activate with:"
	@echo "  conda activate $(CONDA_ENV)"

# Data download and preparation
data:
	$(PYTHON) scripts/download_data.py --dataset german --create-sample
	@echo "Data ready in data/"

data-all:
	$(PYTHON) scripts/download_data.py --dataset all --create-sample
	@echo "All datasets ready in data/"

# Run audit pipeline
run:
	$(PYTHON) scripts/run_audit.py $(ARGS)

run-german:
	$(PYTHON) scripts/run_audit.py --dataset german --mitigation all --out-dir reports/german

run-german-age:
	$(PYTHON) scripts/run_audit.py --dataset german --protected-attr age --mitigation all --out-dir reports/german_age

# Testing
test:
	@echo "Running smoke tests..."
	pytest tests/ -v --tb=short

test-quick:
	@echo "Running quick integration test..."
	$(PYTHON) scripts/download_data.py --dataset german --create-sample --sample-size 200
	$(PYTHON) scripts/run_audit.py \
		--dataset german \
		--data-path data/sample_german_credit.csv \
		--sample-size 200 \
		--mitigation all \
		--out-dir reports/test
	@echo "Quick test passed!"

test-ci:
	@echo "Running CI tests with minimal data..."
	$(PYTHON) scripts/download_data.py --dataset german --create-sample --sample-size 500
	pytest tests/ -v --tb=short
	$(PYTHON) scripts/run_audit.py \
		--dataset german \
		--data-path data/sample_german_credit.csv \
		--sample-size 500 \
		--mitigation all \
		--out-dir reports/ci
	@echo "CI tests passed!"

# Code quality
lint:
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

# Cleanup
clean:
	rm -rf reports/*.csv reports/*.md
	rm -rf data/*.csv data/raw/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf .coverage htmlcov/

clean-all: clean
	rm -rf reports/

# Keep reports directory with gitkeep
reports/.gitkeep:
	mkdir -p reports
	touch reports/.gitkeep
