#!/bin/bash

function check_env_name() {
    if [ $1 = "<your_env_name>" ]; then
        echo $'Error:\n'
        echo $'Please, edit the environment.yml and replace \"<your_env_name>\" by a name for your environment!\n'
        exit
    fi
}

function get_env_name {
    ENV_NAME=$(grep 'name' environment.yml | sed 's/name: //g' | sed 's/\r//g')
    check_env_name $ENV_NAME
    echo $ENV_NAME
}

# Get env name from environment.yml
# "<your_env_name>" must be replaced in environment.yml
ENV_NAME=$(get_env_name)

# Update conda, if required
echo "Updating conda..."
conda update -n base -c defaults conda -y

# # Create conda env according to specs in environment.yml
echo 'Installing packages...'
conda env create -f environment.yml

# Install ipykernel
source activate $ENV_NAME
PYTHON_VERSION=$(python -V)
DISPLAY_NAME="$PYTHON_VERSION ($ENV_NAME)"

echo "Installing ipykernel for:"
echo "      - environment name = $ENV_NAME"
echo "      - python version = $PYTHON_VERSION"
echo "      - display-name = $DISPLAY_NAME"
python -m ipykernel install --user --name $ENV_NAME --display-name "$DISPLAY_NAME"

# Remove unused packages (if any) and tarballs
echo "Cleaning everything..."
conda clean --yes --all
