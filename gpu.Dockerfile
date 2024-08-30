# FROM ros:humble-ros-base-jammy

# FROM ghcr.io/nvidia/jax:jax
# RUN apt-get update
# RUN apt-get install -y gedit
# RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
# RUN python3 -m pip install PyQt5
# RUN python3 -m pip install matplotlib
# RUN python3 -m pip install gpjax
# RUN pip3 uninstall gpjax -y

# ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get install -y libqt5gui5
# RUN apt-get install -y texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng cm-super
# FROM hardikparwana/cuda12-ubuntu22:ros2
FROM nvcr.io/nvidia/jax:23.08-py3

# Set environment variables to avoid interactive prompts during installation and optimize Python environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    # POETRY_HOME='/root/.local/bin' \
    # POETRY_NO_INTERACTION=1 \
    # POETRY_VIRTUALENVS_CREATE=false \
    YOUR_ENV=default_value \
    PATH="/root/.local/bin:${PATH}"

# Version pinning as ARGs for easier updates
# ARG PYTHON_VERSION=3.10
# ARG POETRY_VERSION=1.7.1

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
# RUN python3.10 -m pip install poetry==$POETRY_VERSION

# Expose the port that Jupyter Notebook will run on
EXPOSE 8888

WORKDIR /home/cbfkit

# Copy the project files
# COPY pyproject.toml poetry.lock ./

# Set the PYTHONPATH to include /home and project directories
ENV PYTHONPATH="/home:/home/cbfkit:/home/cbfkit/src:${PYTHONPATH}"

# Project initialization and conditionally install cvxopt if on x86 architecture
# RUN poetry install --no-interaction && \
#     if [ "$(uname -m)" = "x86_64" ]; then poetry add cvxopt; fi

# RUN python3 -m pip install jax[cuda12]
RUN python3 -m pip install numpy matplotlib notebook PyQt5
RUN python3 -m pip install kvxopt>=1.3.2.0
RUN python3 -m pip install cmake>=3.28.1
RUN python3 -m pip install cython>=3.0.8
RUN python3 -m pip install pyyaml>=6.0.1
RUN python3 -m pip install setuptools>=69.0.3
RUN python3 -m pip install black>=23.12.1
RUN python3 -m pip install mypy>=1.8.0
RUN python3 -m pip install jaxopt>=0.8.3
RUN python3 -m pip install jupyter>=1.0.0
RUN python3 -m pip install control>=0.9.4
RUN python3 -m pip install matplotlib>=3.8.2
RUN python3 -m pip install pandas>=2.1.4
RUN python3 -m pip install cvxpy>=1.4.1
RUN python3 -m pip install cvxpylayers>=0.1.6
RUN python3 -m pip install casadi>=3.6.4
RUN python3 -m pip install tqdm>=4.66.2
RUN if [ "$(uname -m)" = "x86_64" ]; then python3 -m pip install cvxopt; fi

RUN apt-get update && apt-get install -y python3-tk
RUN python3 -m pip install PyQt5
RUN apt-get install -y ffmpeg
RUN apt-get install -y '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev
RUN apt-get install -y libqt5gui5
RUN apt-get install -y texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra dvipng cm-super
RUN apt install -y vim