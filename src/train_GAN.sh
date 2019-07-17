#!/bin/bash
ROOT_DIR=/home/guanyu/guanyu/handoff
DATA_PATH=$ROOT_DIR/data

iteration=1
bnd_type=gas
setting=match 

# Train GAN and output phoneme posterior
cd GAN-based-model

python main.py --mode train --cuda_id 0 --bnd_type $bnd_type --setting $setting \
               --data_dir $DATA_PATH \
               --save_dir $DATA_PATH/save/${bnd_type}_${setting} \
               --config "./config.yaml" 

cd ../ 

# WFST decode the phoneme sequences
cd WFST-decoder
python scripts/decode.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                         --data_path $DATA_PATH --jobs 8
python scripts/decode.py --bnd_type $bnd_type --set_type train --lm_type $setting \
                         --data_path $DATA_PATH --jobs 8
cd ../

# Evalution
python eval_per.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                   --data_path $DATA_PATH | tee $DATA_PATH/timit/result/$bnd_type-$setting-test.log
python eval_per.py --bnd_type $bnd_type --set_type train --lm_type $setting \
                   --data_path $DATA_PATH | tee $DATA_PATH/timit/result/$bnd_type-$setting-train.log
