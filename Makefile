.DEFAULT_GOAL := help

DATA_RAW  := data/raw/adult.csv
DATA_CLEAN := data/processed/clean_adult.csv
TRAIN_PREP := data/processed/preprocessed_train.csv

EDA_FIGS := \
	artifacts/figures/age_hist.png \
	artifacts/figures/age_density.png \
	artifacts/figures/marital_status_freq.png \
	artifacts/figures/race_freq.png \
	artifacts/figures/wc_freq.png \
	artifacts/figures/edu_freq.png \
	artifacts/figures/nc_freq.png \
	artifacts/figures/corr_heatmap.png \
	artifacts/figures/pred_class_dist.png

MODEL_ARTIFACTS := \
	artifacts/tables/cv_summary.csv \
	artifacts/tables/log_reg_coefficients.csv \
	artifacts/tables/hpo_results.csv \
	artifacts/figures/cv_summary.png \
	artifacts/figures/log_reg_coefficients.png \
	artifacts/figures/cv_summary_table.png \
	artifacts/figures/log_reg_coefficients_table.png \
	artifacts/figures/hpo_results_table.png

REPORT_DEPS := \
	reports/income-prediction.qmd \
	artifacts/figures/pred_class_dist.png \
	artifacts/figures/corr_heatmap.png \
	artifacts/figures/age_density.png \
	artifacts/figures/edu_freq.png \
	artifacts/figures/race_freq.png \
	artifacts/figures/marital_status_freq.png \
	artifacts/figures/nc_freq.png \
	artifacts/figures/log_reg_coefficients.png \
	artifacts/tables/cv_summary.csv

.PHONY: help all report data split preprocess validate eda model clean \
	cl env build up stop run download-data clean-data split-data preprocess-data data-validate \
	model-train model-reuse run-all-py

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

all: data validate split preprocess eda model report ## Run full pipeline and render report

report: docs/index.html ## Render Quarto report/site

docs/index.html: $(REPORT_DEPS)
	quarto render

data: $(DATA_CLEAN) ## Download raw data and clean it

$(DATA_RAW): scripts/download_data.py
	python scripts/download_data.py --write_to data/raw --target-name adult.csv

$(DATA_CLEAN): scripts/clean_data.py $(DATA_RAW)
	python scripts/clean_data.py --input-path $(DATA_RAW) --output-path $(DATA_CLEAN)

split: data/processed/train.csv data/processed/test.csv ## Split clean data into train/test

data/processed/train.csv data/processed/test.csv: scripts/split_data.py $(DATA_CLEAN)
	python scripts/split_data.py --input_dir="$(DATA_CLEAN)" --out_dir="data/processed/"

preprocess: data/processed/preprocessed_train.csv data/processed/preprocessed_test.csv ## Preprocess train and test data

data/processed/preprocessed_train.csv: scripts/preprocess_data.py data/processed/train.csv
	python scripts/preprocess_data.py --input_dir="data/processed/train.csv" --out_dir="data/processed/"

data/processed/preprocessed_test.csv: scripts/preprocess_data.py data/processed/test.csv
	python scripts/preprocess_data.py --input_dir="data/processed/test.csv" --out_dir="data/processed/"

validate: ## Run data validation tests (non-failing)
	python scripts/validations/data_validation_tests.py --data $(DATA_CLEAN) --no-raise-errors || true

eda: $(EDA_FIGS) ## Run exploratory data analysis and generate figures

$(EDA_FIGS): scripts/eda.py $(DATA_CLEAN)
	python scripts/eda.py --input_dir="$(DATA_CLEAN)" --out_dir="artifacts/figures/"

model: $(MODEL_ARTIFACTS) ## Train models and generate artifacts

$(MODEL_ARTIFACTS): scripts/modeling.py $(TRAIN_PREP)
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --log-reg-iter 30 --svm-iter 20

clean: ## Remove processed data and figures (keeps directories)
	rm -rf data/processed
	rm -rf artifacts/figures
	mkdir -p artifacts/figures

# --- legacy aliases (kept for convenience) ---

download-data: ## Download raw dataset only
	python scripts/download_data.py

clean-data: ## Clean raw dataset only
	python scripts/clean_data.py

split-data: ## Split clean data into train/test
	python scripts/split_data.py --input_dir="data/processed/clean_adult.csv" --out_dir="data/processed/"

preprocess-data: ## Preprocess train and test data
	python scripts/preprocess_data.py --input_dir="data/processed/train.csv" --out_dir="data/processed/"
	python scripts/preprocess_data.py --input_dir="data/processed/test.csv" --out_dir="data/processed/"


model-train: ## Train models with hyperparameter search
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --log-reg-iter 30 --svm-iter 20

model-reuse: ## Reuse tuned models to regenerate tables/figures
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --skip-hpo

analysis: ## Run full Python pipeline (no Quarto)
	$(MAKE) data validate split preprocess eda model

# --- docker / env targets ---

cl: ## Create conda-lock for multiple platforms
	conda-lock lock --file environment.yml -p linux-64 -p osx-64 -p osx-arm64 -p win-64 -p linux-aarch64

env: ## Recreate conda environment from lock file
	conda env remove -n 522-milestone -y || true
	conda-lock install -n 522-milestone conda-lock.yml

build: ## Build Docker image
	docker build -t 522-milestone --file Dockerfile .

run: ## Alias for up
	$(MAKE) up

up: ## Start docker-compose services
	$(MAKE) stop
	docker-compose up -d

stop: ## Stop docker-compose services
	docker-compose stop
