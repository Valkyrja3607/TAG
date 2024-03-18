PROJECT_DIR = $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

DOCKER_IMAGE = tag
DATASET_DIR = $(PWD)/../../datasets

.PHONY: all
all: help
	# Do nothing

.PHONY: help
help: ## This is help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-$(HELP_WIDTH)s\033[0m %s\n", $$1, $$2}'

.PHONY: clean
clean: ## Clean untracked files.
	git clean -dfx

.PHONY: build
build: ## Docker build
	docker build -t $(DOCKER_IMAGE) -f docker/Dockerfile .

.PHONY: download_database
download_database: ## Download database
	docker run -it \
		-v $(PWD):/workspace/$(DOCKER_IMAGE) \
		-v $(DATASET_DIR):/workspace/datasets \
		--name docker-example \
		--rm \
		--shm-size=20g \
		-w /workspace/$(DOCKER_IMAGE) \
		--gpus all \
		$(DOCKER_IMAGE) python -c "from scripts.retrieve_utils.src.models.components.retrieval import download_retrieval_databases; download_retrieval_databases()"

.PHONY: run
run: ## Docker run
	docker run -it \
		-v $(PWD):/workspace/$(DOCKER_IMAGE) \
		-v $(DATASET_DIR):/workspace/datasets \
		--name docker-example \
		--rm \
		--shm-size=20g \
		-w /workspace/$(DOCKER_IMAGE) \
		--gpus all \
		$(DOCKER_IMAGE) python scripts/run.py

.PHONY: bash
bash: ## Enter docker image
	docker run -it \
		-v $(PWD):/workspace/$(DOCKER_IMAGE) \
		-v $(DATASET_DIR):/workspace/datasets \
		--name docker-example \
		--rm \
		--shm-size=20g \
		-w /workspace/$(DOCKER_IMAGE) \
		--gpus all \
		$(DOCKER_IMAGE) bash
