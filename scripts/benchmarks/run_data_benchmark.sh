set -e

SCRIPT_PATH=$(dirname $(realpath $0))
DATA_BENCHMARK_PATH=${SCRIPT_PATH}/data
CODE_PATH=$(realpath ${SCRIPT_PATH}/../../)

DATASET_PATH=$(realpath ${CODE_PATH}/../data/training/mvsec)
PREPROCESSED_DATASET_PATH=${DATASET_PATH}/preprocessed/1_1_1
QUANTIZED_DATASET_PATH=${DATASET_PATH}/quantized/1_1_1

CACHE_SIZE=2

MODEL_PATH=$(mktemp -d)

COMMON_ARGS=$(echo "-m ${MODEL_PATH} \
             --flownet_path ${CODE_PATH}/EV_OFlowNet \
             --suffix-length 1 \
             --prefix-length 1 \
             --min-sequence-length 3 \
             --max-sequence-length 3 \
             -d cuda:0 \
             -bs 8 \
             -mbs 8 \
             --optimizer ADAM \
             --checkpointing_interval 1000 \
             --num_workers 2 \
             --event-representation-depth 3 \
             --allow-obsolete-code \
             --allow-arguments-change" | tr -s " ")

echo "Raw data"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  ${COMMON_ARGS}

echo "Preprocessed dataset without cache"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
  ${COMMON_ARGS}

CACHE_DIR=$(mktemp -d)
echo "Preprocessed dataset with cache but processing only once"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
  --cache-dir ${CACHE_DIR} \
  --cache-size ${CACHE_SIZE} \
  --process-only-once \
  ${COMMON_ARGS}
rm -rf ${CACHE_DIR}

CACHE_DIR=$(mktemp -d)
echo "Preprocessed dataset with cache with multiple passes over cached data"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  --preprocessed-dataset-path $PREPROCESSED_DATASET_PATH \
  --cache-dir ${CACHE_DIR} \
  --cache-size ${CACHE_SIZE} \
  ${COMMON_ARGS}
rm -rf ${CACHE_DIR}

echo "Quantized dataset without cache"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  --preprocessed-dataset-path ${QUANTIZED_DATASET_PATH} \
  --ev_images \
  ${COMMON_ARGS}

CACHE_DIR=$(mktemp -d)
echo "Quantized dataset with cache but processing only once"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  --preprocessed-dataset-path ${QUANTIZED_DATASET_PATH} \
  --ev_images \
  --cache-dir ${CACHE_DIR} \
  --cache-size ${CACHE_SIZE} \
  --process-only-once \
  ${COMMON_ARGS}
rm -rf ${CACHE_DIR}

CACHE_DIR=$(mktemp -d)
echo "Quantized dataset with cache with multiple passes over cached data"
python3 ${DATA_BENCHMARK_PATH}/profile_dataloader.py \
  --preprocessed-dataset-path ${QUANTIZED_DATASET_PATH} \
  --ev_images \
  --cache-dir ${CACHE_DIR} \
  --cache-size ${CACHE_SIZE} \
  ${COMMON_ARGS}
rm -rf ${CACHE_DIR}

rm -rf ${MODEL_PATH}
