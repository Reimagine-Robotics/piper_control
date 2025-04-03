#!/bin/bash

# Script to create a conda environment with:
# -  Python 3.10
# -  ipykernel
# -  piper_control (editable install)
#
# Usage:
#   ./conda_env.sh
# This script assumes that conda is already installed and available in the PATH.
# Check if conda is installed with:
#   conda --version
#
# This script can be used to create a conda env for local development or testing.

ENV_NAME="piper_control_env"

# Create the conda environment
if conda create --name "$ENV_NAME" python=3.10 --yes; then
  echo "Conda environment '$ENV_NAME' with Python 3.10 created successfully."
else
  echo "Failed to create conda environment '$ENV_NAME'."
  exit 1
fi

# Initialize conda.
if eval "$(conda shell.bash hook)"; then
  echo "Conda initialized successfully."
else
  echo "Failed to initialize conda."
  exit 1
fi

# Activate the conda environment.
if conda activate "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' activated successfully."
else
  echo "Failed to activate conda environment '$ENV_NAME'."
  exit 1
fi

# Install ipykernel in the conda environment.
if conda install ipykernel --yes; then
  echo "ipykernel installed successfully in conda environment '$ENV_NAME'."
else
  echo "Failed to install ipykernel in conda environment '$ENV_NAME'."
  exit 1
fi

# Install pyink in the conda environment so code formatting works in local development.
if pip install pyink; then
  echo "pyink installed successfully in conda environment '$ENV_NAME'."
else
  echo "Failed to install pyink in conda environment '$ENV_NAME'."
  exit 1
fi

# Install this project, editable.
if pip install -e .; then
  echo "piper_control installed successfully in conda environment '$ENV_NAME'."
else
  echo "Failed to install piper_control in conda environment '$ENV_NAME'."
  exit 1
fi
