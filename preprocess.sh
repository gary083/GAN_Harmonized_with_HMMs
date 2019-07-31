#!/bin/bash

. ./config.sh
. ./cmd.sh
. ./path.sh

# Process TIMIT
bash src/process-timit/process_timit.sh $ROOT_DIR $TIMIT_DIR

# Data preparation for KALDI(WFST)
cd src/WFST-decoder
[ -d data ] || mkdir $feat_dir

bash scripts/preprocess.sh match $DATA_PATH 
bash scripts/preprocess.sh nonmatch $DATA_PATH 
cd ../
