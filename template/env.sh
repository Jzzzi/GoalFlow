#!/bin/bash

# You can copy this file to the root directory of the project
# so that you can run `source env.sh` to activate the conda environment.
# This is a template file, please modify the path according to your own environment.

# === Conda Environment ===
source /baai-cwm-1/baai_cwm_ml/cwm/huanang.gao/env/etc/profile.d/conda.sh
conda activate /baai-cwm-1/baai_cwm_ml/algorithm/zongzheng.zhang/.conda/envs/go

# === NavSim Enviroment ===
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$PROJECT_DIR/dataset/maps"
export NAVSIM_EXP_ROOT="$PROJECT_DIR/exp"
export NAVSIM_DEVKIT_ROOT="$PROJECT_DIR/navsim"
export OPENSCENE_DATA_ROOT="$PROJECT_DIR/dataset"

