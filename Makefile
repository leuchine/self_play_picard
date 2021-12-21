GIT_HEAD_REF := e37020b6eee18bff865d9d2ba852bd636f3ed777

BASE_IMAGE := pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

DEV_IMAGE_NAME := text-to-sql-dev
TRAIN_IMAGE_NAME := text-to-sql-train
EVAL_IMAGE_NAME := text-to-sql-eval

BUILDKIT_IMAGE := tscholak/text-to-sql-buildkit:buildx-stable-1
BUILDKIT_BUILDER ?= buildx-local
BASE_DIR := $(shell pwd)

.PHONY: init-buildkit
init-buildkit:
	docker buildx create \
		--name buildx-local \
		--driver docker-container \
		--driver-opt image=$(BUILDKIT_IMAGE),network=host \
		--use

.PHONY: del-buildkit
del-buildkit:
	docker buildx rm buildx-local

.PHONY: build-thrift-code
build-thrift-code:
	thrift1 --gen mstch_cpp2 picard.thrift
	thrift1 --gen mstch_py3 picard.thrift
	cd gen-py3 && python setup.py build_ext --inplace

.PHONY: build-picard-deps
build-picard-deps:
	cabal update
	thrift-compiler --hs --use-hash-map --use-hash-set --gen-prefix gen-hs -o . picard.thrift
	patch -p 1 -N -d third_party/hsthrift < ./fb-util-cabal.patch || true
	cd third_party/hsthrift \
		&& make THRIFT_COMPILE=thrift-compiler thrift-cpp thrift-hs
	cabal build --only-dependencies lib:picard

.PHONY: build-picard
build-picard:
	cabal install --overwrite-policy=always --install-method=copy exe:picard

.PHONY: build-dev-image
build-dev-image:
	ssh-add
	docker buildx build \
		--builder $(BUILDKIT_BUILDER) \
		--ssh default=$(SSH_AUTH_SOCK) \
		-f Dockerfile \
		--tag tscholak/$(DEV_IMAGE_NAME):$(GIT_HEAD_REF) \
		--tag tscholak/$(DEV_IMAGE_NAME):cache \
		--tag tscholak/$(DEV_IMAGE_NAME):devcontainer \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--target dev \
		--cache-from type=registry,ref=tscholak/$(DEV_IMAGE_NAME):cache \
		--cache-to type=inline \
		--push \
		git@github.com:ElementAI/picard#$(GIT_HEAD_REF)

.PHONY: pull-dev-image
pull-dev-image:
	docker pull tscholak/$(DEV_IMAGE_NAME):$(GIT_HEAD_REF)

.PHONY: build-train-image
build-train-image:
	ssh-add
	docker buildx build \
		--builder $(BUILDKIT_BUILDER) \
		--ssh default=$(SSH_AUTH_SOCK) \
		-f Dockerfile \
		--tag tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		--tag tscholak/$(TRAIN_IMAGE_NAME):cache \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--target train \
		--cache-from type=registry,ref=tscholak/$(TRAIN_IMAGE_NAME):cache \
		--cache-to type=inline \
		--push \
		git@github.com:ElementAI/picard#$(GIT_HEAD_REF)

.PHONY: pull-train-image
pull-train-image:
	docker pull tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF)

.PHONY: build-eval-image
build-eval-image:
	ssh-add
	docker buildx build \
		--builder $(BUILDKIT_BUILDER) \
		--ssh default=$(SSH_AUTH_SOCK) \
		-f Dockerfile \
		--tag tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF) \
		--tag tscholak/$(EVAL_IMAGE_NAME):cache \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--target eval \
		--cache-from type=registry,ref=tscholak/$(EVAL_IMAGE_NAME):cache \
		--cache-to type=inline \
		--push \
		git@github.com:ElementAI/picard#$(GIT_HEAD_REF)

.PHONY: pull-eval-image
pull-eval-image:
	docker pull tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF)

.PHONY: train
train: pull-train-image
	mkdir -p -m 777 train
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
		-m8g \
		--rm \
		--runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=5 \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/train,target=/train \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_train_text2sql.py configs/train.json"

.PHONY: bash
bash: pull-train-image
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	mkdir -p -m 777 train
	docker run \
		-m8g \
		--rm \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/train,target=/train \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash


.PHONY: train_cosql
train_cosql: pull-train-image
	mkdir -p -m 777 train_cosql
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
		-m8g \
		--rm \
		--runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=1,2,3,4 \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/train_cosql,target=/train_cosql \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_train_text2sql.py configs/train_cosql.json"

.PHONY: train_cosql_self_play
train_cosql_self_play: pull-train-image
	mkdir -p -m 777 train_cosql_self_play
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
		-m8g \
		--rm \
		--runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/train_cosql_self_play,target=/train_cosql_self_play \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_train_text2sql.py configs/train_cosql.json True"


.PHONY: train_sparc
train_sparc: pull-train-image
	mkdir -p -m 777 train_sparc
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
		-m8g \
		--rm \
		--runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=5 \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/train_sparc,target=/train_sparc \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_train_text2sql.py configs/train_sparc.json"


.PHONY: train_sql2text_cosql
train_sql2text_cosql: pull-train-image
	mkdir -p -m 777 train_sql2text_cosql
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
		-m8g \
		--rm \
		--runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=5 \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/train_sql2text_cosql,target=/train_sql2text_cosql \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(TRAIN_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "pip install sacrebleu;python seq2seq/run_train_sql2text.py configs/train_sql2text_cosql.json"

.PHONY: eval
eval: pull-eval-image
	mkdir -p -m 777 eval
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
        -m8g \
		--rm \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/eval,target=/eval \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_train_text2sql.py configs/eval.json"

.PHONY: eval_cosql
eval_cosql: pull-eval-image
	mkdir -p -m 777 eval_cosql
	mkdir -p -m 777 transformers_cache
	mkdir -p -m 777 wandb
	docker run \
        -m8g \
		--rm \
		--user 13011:13011 \
		--mount type=bind,source=$(BASE_DIR)/eval_cosql,target=/eval_cosql \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/wandb,target=/app/wandb \
		tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_train_text2sql.py configs/eval_cosql.json"

.PHONY: self_play_cosql
self_play_cosql: pull-eval-image
	mkdir -p -m 777 database
	mkdir -p -m 777 transformers_cache
	docker run \
		-m8g \
		--rm \
		--user 13011:13011 \
		-p 8000:8000 \
		--runtime=nvidia \
		-e NVIDIA_VISIBLE_DEVICES=0 \
		--mount type=bind,source=$(BASE_DIR)/database,target=/database \
		--mount type=bind,source=$(BASE_DIR)/gazp-main,target=/gazp-main \
		--mount type=bind,source=$(BASE_DIR)/train_cosql,target=/train_cosql \
		--mount type=bind,source=$(BASE_DIR)/train_sql2text_cosql,target=/train_sql2text_cosql \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		--mount type=bind,source=$(BASE_DIR)/seq2seq,target=/app/seq2seq \
		tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/run_self_play.py configs/self_play_cosql.json"

.PHONY: serve
serve: pull-eval-image
	mkdir -p -m 777 database
	mkdir -p -m 777 transformers_cache
	docker run \
		-it \
		--rm \
		--user 13011:13011 \
		-p 8000:8000 \
		--runtime=nvidia \
		--mount type=bind,source=$(BASE_DIR)/database,target=/database \
		--mount type=bind,source=$(BASE_DIR)/train,target=/train \
		--mount type=bind,source=$(BASE_DIR)/transformers_cache,target=/transformers_cache \
		--mount type=bind,source=$(BASE_DIR)/configs,target=/app/configs \
		tscholak/$(EVAL_IMAGE_NAME):$(GIT_HEAD_REF) \
		/bin/bash -c "python seq2seq/serve_seq2seq.py configs/serve.json"

# pip install /app/seq2seq/portalocker-2.3.2-py2.py3-none-any.whl; pip install /app/seq2seq/colorama-0.4.4-py2.py3-none-any.whl; pip install /app/seq2seq/tabulate-0.8.9-py3-none-any.whl ;python3 -m pip install /app/seq2seq/sacrebleu-2.0.0-py3-none-any.whl;
