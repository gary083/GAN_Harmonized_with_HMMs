# !/bin/bash

### Setting for Kaldi
. ./cmd.sh
. ./path.sh


### Experimental Setting 
. config

if [ $bnd_type == orc ]; then
  total_iter=1
else
  total_iter=3
fi

### Preprocess TIMIT
# bash preprocess.sh


### Training Process
cd src

for iteration in $(seq 1 $total_iter); do
  ### train GAN model and get transcriptions
  bash train_GAN.sh $iteration || exit 0

  ### train HMM and get new boundaries
  bash train_HMM.sh $iteration || exit 0
done


# iteration=2
# bash train_GAN.sh $iteration
# bash train_HMM.sh $iteration