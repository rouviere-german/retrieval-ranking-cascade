# Linux/CI convenience targets. Every target delegates to tasks.py, which is the single source
# of truth and also runs on Windows: `uv run python tasks.py <target>`.

UV ?= uv
TASK := $(UV) run python tasks.py
IMAGE ?= retrieval-ranking-cascade:local

.PHONY: install lint format typecheck test bench quality load demo bandit pip-audit audit baseline check full docker-build compose-up train

install:
	$(UV) sync --frozen

lint:
	$(TASK) lint

format:
	$(TASK) format

typecheck:
	$(TASK) typecheck

test:
	$(TASK) test

bench:
	$(TASK) bench

quality:
	$(TASK) quality

load:
	$(TASK) load

demo:
	$(TASK) demo

bandit:
	$(TASK) bandit

pip-audit:
	$(TASK) pip-audit

audit:
	$(TASK) audit

baseline:
	$(TASK) baseline

check:
	$(TASK) check

full:
	$(TASK) full

docker-build:
	docker build -t $(IMAGE) .

train:
	docker compose run --rm train

compose-up:
	docker compose up --build app
