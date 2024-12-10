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
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    make \
    build-essential \
    software-properties-common \
    curl \
    ssh \
    git \
    # python3-pip \
    # python3-dev \
    libopenblas-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    # && apt-get install -y python3.10 python3.10-dev python3.10-venv \
    # && apt-get clean \
    # && rm -rf /var/lib/apt/lists/* \
    && apt-get install -y ffmpeg
    # && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

WORKDIR /home/cbfkit

# Copy the project files
COPY pyproject.toml ./

# Set the PYTHONPATH to include /home and project directories
ENV PYTHONPATH="/home:/home/cbfkit:/home/cbfkit/src:${PYTHONPATH}"

# Install dependencies using uv
RUN uv sync --no-install-project 

# Source the ROS 2 environment for all users when starting a shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Set the safe directory to /home/cbfkit
RUN git config --global --add safe.directory /home/cbfkit

# Activate the environment in every shell session
RUN echo "source .venv/bin/activate" >> /root/.bashrc