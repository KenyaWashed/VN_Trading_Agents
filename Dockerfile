# Sealed Evaluation Sandbox Container
# Enforces no-network execution (--network=none at runtime) and read-only human-owned evaluator.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Create unprivileged agent runner user
RUN groupadd -r sandbox && useradd -r -g sandbox -m agent_runner

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY experiments/ ./experiments/
COPY tests/ ./tests/
COPY paper/ ./paper/

RUN pip install --no-cache-dir -e ".[dev]"

# Lock human-owned evaluator, sandbox, data, and configs as read-only for agent_runner
RUN chmod -R a-w /workspace/src/vn_rsi/sandbox \
                 /workspace/src/vn_rsi/eval \
                 /workspace/src/vn_rsi/data \
                 /workspace/src/vn_rsi/dsl \
                 /workspace/configs && \
    mkdir -p /workspace/logs /workspace/data && \
    chown -R agent_runner:sandbox /workspace/logs

USER agent_runner

# Run container with: docker run --rm --network=none --read-only --tmpfs /tmp -v $(pwd)/logs:/workspace/logs vn-rsi-sandbox
CMD ["pytest", "-q"]
