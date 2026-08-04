FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and dependencies
COPY requirements.txt /app/
RUN pip install uv && uv pip install --system -r requirements.txt

# Map user to host UID/GID for permission consistency
ARG UID
ARG GID
RUN groupadd -o -g $GID myuser && useradd -u $UID -g $GID -m myuser
RUN chown -R myuser:myuser /app
USER myuser

ENV HF_HOME="/app/cache/huggingface"

# The container will always run this script on start
ENTRYPOINT ["/bin/bash"]