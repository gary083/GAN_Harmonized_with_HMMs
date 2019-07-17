#!/bin/bash
ROOT_DIR=/home/guanyu/guanyu/handoff
TIMIT_DIR=/home/guanyu/guanyu/timit_data
DATA_PATH=$ROOT_DIR/data

# Process TIMIT
bash process-timit/process_timit.sh $ROOT_DIR $TIMIT_DIR

# Data preparation for KALDI(WFST)
cd WFST-decoder
mkdir data
bash scripts/preprocess.sh match $DATA_PATH 
bash scripts/preprocess.sh nonmatch $DATA_PATH 
cd ../
