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

.PHONY: data-validate
data-validate: ## run data validation tests
	python ./src/validations/data_validation_tests.py


.PHONY: model-train
model-train: ## Train with hyperparameter search and save tuned pickles/figures
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --log-reg-iter 50 --svm-iter 30

.PHONY: model-reuse
model-reuse: ## Reuse existing tuned pickles to regenerate tables/figures (skip HPO)
	python scripts/modeling.py --data-dir data --artifacts-dir artifacts \
		--cv-folds 5 --top-n 10 --skip-hpo
