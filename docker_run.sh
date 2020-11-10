#!/bin/bash
if [[ -d /scratch ]]; then
    SCRATCH_PATH=/scratch/of
else
    SCRATCH_PATH=$(realpath $(pwd)/..)
fi
DATA_PATH=$SCRATCH_PATH/data/training
MODELS_PATH=$SCRATCH_PATH/models

set -x -e 

nvidia-docker run -it \
  --net=host \
  --volume="$DATA_PATH:/data/training" \
  --volume="$(pwd)/data/info:/data/info" \
  --volume="$(pwd)/../data/raw:/data/raw" \
  --volume="$MODELS_PATH:/models" \
  --security-opt apparmor:unconfined \
  --ipc=host \
  -e INSIDE_DOCKER=1 \
  of:train /bin/bash
