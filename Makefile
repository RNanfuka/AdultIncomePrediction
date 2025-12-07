.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: all
all: ## runs the targets: cl, env, build
	make cl
	make env
	make build

.PHONY: cl
cl: ## create conda lock for multiple platforms
	# the linux-aarch64 is used for ARM Macs using linux docker container, but not compatible with quarto.
	conda-lock lock \
		--file environment.yml \
		-p linux-64 \
		-p osx-64 \
		-p osx-arm64 \
		-p win-64 \
		-p linux-aarch64

.PHONY: env
env: ## remove previous and create environment from lock file
	# remove the existing env, and ignore if missing
	conda env remove -n 522-milestone -y || true
	conda-lock install -n 522-milestone conda-lock.yml

.PHONY: build
build: ## build the docker image from the Dockerfile
	docker build -t 522-milestone --file Dockerfile .

.PHONY: run
run: ## alias for the up target
	make up

.PHONY: up
up: ## stop and start docker-compose services
	# by default stop everything before re-creating
	make stop
	docker-compose up -d

.PHONY: stop
stop: ## stop docker-compose services
	docker-compose stop

.PHONY: download-data
download-data:
	python scripts/download_data.py

.PHONY: clean-data
clean-data:
	python scripts/clean_data.py

.PHONY: split-data
split-data:
	python scripts/split_data.py --input_dir="data/processed/clean_adult.csv" --out_dir="data/processed/"

.PHONY: preprocess-data
preprocess-data:
	python scripts/preprocess_data.py --input_dir="data/processed/train.csv" --out_dir="data/processed/"
	python scripts/preprocess_data.py --input_dir="data/processed/test.csv" --out_dir="data/processed/"

.PHONY: data-validate
data-validate: ## run data validation tests
	python scripts/validations/data_validation_tests.py --data data/processed/clean_adult.csv --no-raise-errors || true

.PHONY: eda
eda: ## run data validation tests
	python scripts/eda.py --input_dir="data/clean.csv" --out_dir="artifacts/figures/"

.PHONY: model-train
model-train: ## Train with hyperparameter search and save tuned pickles/figures
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --log-reg-iter 50 --svm-iter 30

.PHONY: model-reuse
model-reuse: ## Reuse existing tuned pickles to regenerate tables/figures (skip HPO)
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --skip-hpo

.PHONY: run-all-py
run-all-py: ## Run all python scripts
	make download-data
	make clean-data
	make data-validate
	make split-data
	make eda
	make preprocess-data
	make model-train
	make model-reuse
