###################################################
FROM python:3.10.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml poetry.lock* /app/

WORKDIR /app

RUN set -e; \
    runtimeDeps=' \
        # to install poetry and make publishing work
        git ssh \
        # to work with opencv \
        libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
        # matplotlib
        libfreetype6-dev libpng-dev \
        '; \
    apt-get update; \
    apt-get install -y --no-install-recommends $runtimeDeps;

RUN set -e; \
    buildDeps=' \
        # to install poetry \
        curl git \
        '; \
    apt-get update; \
    apt-get install -y --no-install-recommends $buildDeps; \
    rm -rf /var/lib/apt/lists/*; \
    curl -sSL https://install.python-poetry.org | python3 - --version 1.4.2; \
    ln -s /root/.local/bin/poetry /bin/poetry; \
    poetry config virtualenvs.create false; \
    # only install if lockfile exists
    [ ! -f "/app/poetry.lock" ] || poetry install --no-root --without dev; \
    apt-get purge -y --auto-remove $buildDeps; \
    rm -r /root/.cache/*

###################################################
FROM builder AS local

RUN apt-get update && apt-get install -y --no-install-recommends \
    # edit pip installed libraries
    vim \
    # ps to investigate / kill hung processes
    procps \
    # make
    build-essential

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=TRUE \
    PYTHONPATH=/app

WORKDIR /app/QCNN