#!/bin/bash
ROOT_DIR=/home/guanyu/guanyu/handoff
DATA_PATH=$ROOT_DIR/data
TIMIT_DIR=/home/guanyu/guanyu/timit_data


iteration=1
bnd_type=orc
setting=match 
jobs=8

gan_prefix=${bnd_type}_iter${iteration}_${setting}_gan
hmm_prefix=${bnd_type}_iter${iteration}_${setting}_hmm

GAN_PATH=$DATA_PATH/save/${gan_prefix}
HMM_PATH=$DATA_PATH/save/${hmm_prefix}

cd unsupervised-HMM

# Prepare data for kaldi scripts(timit)
python local/prepare_data.py --bnd_type $bnd_type --lm_type $setting \
                             --data_path $DATA_PATH \
                             --timit_path $TIMIT_DIR \
                             --iteration $iteration
# Train HMM iteratively 
bash run.sh $DATA_PATH $HMM_PATH $setting $jobs

# Refined phoneme boundaries
bash local/lat_to_phones.sh $HMM_PATH $jobs
cd ../
python get_new_bound.py --bnd_type $bnd_type --iteration $iteration \
                        --prefix $hmm_prefix --data_path $DATA_PATH

Evalution
if [ -e $DATA_PATH/result/${hmm_prefix}.log ]; then 
  rm $DATA_PATH/result/${hmm_prefix}.log
fi 

for dir in mono tri1 tri2 tri3; do
  python eval_per.py --bnd_type $bnd_type --set_type test --lm_type $setting \
                   --data_path $DATA_PATH --prefix $hmm_prefix \
                   --file_name ${dir}_test_output.txt >> $DATA_PATH/result/${hmm_prefix}.log
done
