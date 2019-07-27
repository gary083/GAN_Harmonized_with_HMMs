#!/bin/bash
ROOT_DIR=/home/guanyu/guanyu/handoff
DATA_PATH=$ROOT_DIR/data

iteration=1
bnd_type=orc
setting=match 
jobs=8

prefix=${bnd_type}_iter${iteration}_${setting}_gan

# Train GAN and output phoneme posterior
cd GAN-based-model

python main.py --mode train --cuda_id 0 --bnd_type $bnd_type --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${prefix} \
               --config "./config.yaml"

cd ../ 

# WFST decode the phoneme sequences
cd WFST-decoder
python scripts/decode.py --set_type test --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
python scripts/decode.py --set_type train --lm_type $setting \
                         --data_path $DATA_PATH --prefix $prefix \
                         --jobs $jobs
cd ../

# Evalution
python eval_per.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                   --data_path $DATA_PATH --prefix $prefix \
                   --file_name test_output.txt | tee $DATA_PATH/result/${prefix}.log