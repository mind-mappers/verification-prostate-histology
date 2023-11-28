#!/bin/bash

CONDA_ENV_NAME="generate_data_env"
GENERATE_DATA_CONFIG="./homologous_point_prediction/data_processing/metadata/generate_config.json"

# Create the conda env if it does not exist
ENVS=$(conda env list | awk "{print $CONDA_ENV_NAME}")

if [[ "$ENVS" != *"$CONDA_ENV_NAME"* ]];
    then conda create --name $CONDA_ENV_NAME python=3.9;
fi

conda activate $CONDA_ENV_NAME
pip install -r ./homologous_point_prediction/data_processing/requirements.txt

python ./homologous_point_prediction/data_processing/generate_data.py $GENERATE_DATA_CONFIG
