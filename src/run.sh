# !/bin/bash

# Preprocess TIMIT
bash preprocess.sh
# train GAN model and get transcriptions
bash train_GAN.sh

# train HMM and get new boundaries
bash train_HMM.sh