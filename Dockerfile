FROM ros:humble-ros-base-jammy

# Set environment variables to avoid interactive prompts during installation and optimize Python environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME='/root/.local/bin' \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    YOUR_ENV=default_value \
    PATH="/root/.local/bin:${PATH}"

# Version pinning as ARGs for easier updates
ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=1.7.1

# Update the package list, install necessary packages in one layer, including Python 3.10, CMake, BLAS libraries, and clean up
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    make \
    build-essential \
    software-properties-common \
    curl \
    ssh \
    git \
    python3-pip \
    python3-dev \
    libopenblas-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y python3.10 python3.10-dev python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Poetry using pip and pin the version
RUN python3.10 -m pip install poetry==$POETRY_VERSION

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

WORKDIR /home/cbfkit

# Copy the project files
COPY pyproject.toml poetry.lock ./

# Set the PYTHONPATH to include /home and project directories
ENV PYTHONPATH="/home:/home/cbfkit:/home/cbfkit/src:${PYTHONPATH}"

# Project initialization and conditionally install cvxopt if on x86 architecture
RUN poetry install --no-interaction && \
    if [ "$(uname -m)" = "x86_64" ]; then poetry add cvxopt; fi

RUN apt-get update && apt-get install -y python3-tk
RUN pip3 install PyQt5
RUN apt-get install -y ffmpeg
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN apt-get install -y ffmpeg