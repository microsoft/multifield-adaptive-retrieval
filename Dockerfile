# syntax=docker/dockerfile:1.2.1
# Use an image from https://eng.ms/docs/more/containers-secure-supply-chain/approved-images
# TODO: Switch to a smaller image which doesn't contain the CUDA SDK, CUDNN, and
# other unnecessary large files. conda will download all such files needed for PyTorch again.
# -base (or maybe -runtime) from https://hub.docker.com/r/nvidia/cuda should suffice.
# Tested on an Azure VM with base image: Ubuntu Server 20.04 LTS x64 Gen2
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 as base

# Set up caching for apt
# https://docs.docker.com/engine/reference/builder/#run---mounttypecache
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# https://askubuntu.com/questions/909277/avoiding-user-interaction-with-tzdata-when-installing-certbot-in-a-docker-contai
ENV DEBIAN_FRONTEND=noninteractive

# For python3.10
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Install apt-get dependencies
# ITP (AML K8S): requires sudo, unzip
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    git \
    sudo \
    unzip \
    wget \
    python3.10-dev \
    python3.10-venv \
    python3-dev \
    python-is-python3 \
    python3-pip \
    libsnappy-dev \
    libbz2-dev \
    liblz4-dev

WORKDIR /pkg
RUN git clone https://github.com/usnistgov/trec_eval.git
WORKDIR /pkg/trec_eval
RUN make && make install

# Install Poetry
# Not using the official way of installation, see https://stackoverflow.com/questions/53835198/integrating-python-poetry-with-docker
ARG POETRY_VERSION='1.8.4'
RUN python -m pip install pip "poetry==${POETRY_VERSION}"

# Setup the Poetry environment.
# We do this first to benefit from Docker build layer caching, since dependencies change less often than the code.
WORKDIR /mfar

COPY poetry.lock /mfar/poetry.lock
COPY pyproject.toml /mfar/pyproject.toml

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV="/venv"
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi --no-root
