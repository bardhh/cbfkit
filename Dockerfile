FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    YOUR_ENV=default_value \
    PATH="/root/.local/bin:${PATH}"

ARG PYTHON_VERSION=3.10

# Consolidate all apt operations into a single layer, reduce unnecessary packages, and clean up
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    make \
    build-essential \
    software-properties-common \
    curl \
    ssh \
    git \
    libopenblas-dev \
    ffmpeg \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv in a separate layer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

EXPOSE 8888
WORKDIR /home/cbfkit

# Copy only necessary dependency files first for better layer caching
COPY pyproject.toml ./

ENV PYTHONPATH="/home:/home/cbfkit:/home/cbfkit/src:${PYTHONPATH}"

# Run uv sync after copying dependencies
RUN uv sync --no-install-project

# Source ROS 2 environment in every shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Mark project directory as safe for git
RUN git config --global --add safe.directory /home/cbfkit

# Activate virtual environment in every shell session
RUN echo "source .venv/bin/activate" >> /root/.bashrc