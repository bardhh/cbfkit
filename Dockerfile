FROM ros:humble-ros-base-jammy

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    YOUR_ENV=default_value \
    PATH="/root/.local/bin:${PATH}"

# Version pinning as ARGs for easier updates
ARG PYTHON_VERSION=3.10

# Update the package list, install necessary packages in one layer
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    make \
    build-essential \
    software-properties-common \
    curl \
    ssh \
    git \
    libopenblas-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get install -y ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

WORKDIR /home/cbfkit

# # Copy the project files
COPY pyproject.toml ./

# Set the PYTHONPATH to include /home and project directories
ENV PYTHONPATH="/home:/home/cbfkit:/home/cbfkit/src:${PYTHONPATH}"

# Install dependencies using uv
RUN uv pip install . 

# Source the ROS 2 environment for all users when starting a shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set the safe directory to /home/cbfkit
RUN git config --global --add safe.directory /home/cbfkit