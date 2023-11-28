#!/bin/bash

CONDA_ENV_NAME="generate_data_env"
OUTPUT_DIR_NAME="./outputs/12-17-21|03:08:38"

MODEL_PATH="${OUTPUT_DIR_NAME}/model"
HIST_IMAGE="/data/ur/bukowy/LaViolette_Data/Prostates/1113/7/small_recon_8_pgt_sharp.tiff"
MRI_IMAGE="/data/ur/bukowy/LaViolette_Data/Prostates/1113/7/mri_slice_double_T2_norm.nii"


ENVS=$(conda env list | awk "{print $CONDA_ENV_NAME}")

if [[ "$ENVS" != *"$CONDA_ENV_NAME"* ]];
    then conda create --name $CONDA_ENV_NAME python=3.9;
fi

conda activate $CONDA_ENV_NAME

python ./pipeline.py $MODEL_PATH $HIST_IMAGE $MRI_IMAGE $OUTPUT_DIR_NAME
