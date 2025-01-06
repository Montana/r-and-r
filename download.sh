#!/bin/bash
set -e

BASE_DIR="data"

download_file() {
    local dataset_dir=$1
    local url=$2

    mkdir -p "$BASE_DIR/$dataset_dir"
    echo "Downloading $dataset_dir . . ."
    wget -P "$BASE_DIR/$dataset_dir" "$url"
}

download_file "nq" "https://nlp.stanford.edu/data/nfliu/lost-in-the-middle/nq-open-contriever-msmarco-retrieved-documents.jsonl.gz"
download_file "squad" "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
download_file "hotpotqa" "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"

echo "All downloads complete!"
