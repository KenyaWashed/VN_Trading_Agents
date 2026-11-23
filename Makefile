PYTHON ?= uv run python
PYTEST ?= uv run pytest
RUFF ?= uv run ruff
MYPY ?= uv run mypy

.PHONY: data test lint baseline agent rsi modular train-cloud report verify all

data:
	$(PYTHON) experiments/run_data.py --config configs/data.yaml --market-rules configs/market_rules.yaml

test:
	$(PYTEST) tests/

lint:
	$(RUFF) check src tests experiments paper scripts train_colab_kaggle.py
	$(MYPY) src/vn_rsi

baseline:
	$(PYTHON) experiments/run_baselines.py --sandbox-config configs/sandbox.yaml --data-config configs/data.yaml --market-rules configs/market_rules.yaml

agent:
	$(PYTHON) experiments/run_agent_v0.py --agent-config configs/agent_v0.yaml --llm-config configs/llm.yaml --sandbox-config configs/sandbox.yaml --data-config configs/data.yaml --market-rules configs/market_rules.yaml

rsi:
	$(PYTHON) experiments/run_rsi.py --rsi-config configs/rsi.yaml --agent-config configs/agent_v0.yaml --llm-config configs/llm.yaml --sandbox-config configs/sandbox.yaml --data-config configs/data.yaml --market-rules configs/market_rules.yaml

modular:
	$(PYTHON) experiments/run_modular_rsi.py --modular-config configs/kaggle_modular_rsi.yaml

train-cloud:
	$(PYTHON) train_colab_kaggle.py --preset auto

report:
	$(PYTHON) experiments/build_report.py --logs-dir logs --paper-dir paper
	$(PYTHON) paper/verify_paper.py --paper-dir paper --logs-dir logs

all: data test baseline agent rsi modular report
