# GAN-based model

This is the GAN-based model's implementation of [our paper](#Citation).

## How to use
1. Different setting could be done by adding different arguments.
```
usage: main.py [-h] [--mode MODE] [--model_type MODEL_TYPE]
               [--cuda_id CUDA_ID] [--bnd_type BND_TYPE] [--setting SETTING]
               [--iteration ITERATION] [--aug] [--data_dir DATA_DIR]
               [--save_dir SAVE_DIR] [--config CONFIG]
```
- Example:
```
python main.py --mode train --cuda_id 0 --bnd_type uns 
               --setting match --iteration 2 
               --data_dir [path-to-data]
               --save_dir [path-to-save-directory]
               --config [path-to-config]
```

2. Argument

`mode`: train or test mode (train/test).

`model_type`: supervised / unsupervised model (sup/uns).

`cuda_id`: GPU ids.

`bnd_type`: type of boundaries (orc/uns).

`setting`: source of text (match/nonmatch).

`iteration`: iteration in HMM harmonization.

`data_dir`: directory of your data.

`save_dir`: directory where your save your model and results.

`config`: path to the model config.

## Hyperparameters in `config.yaml`

`gen_hidden_size`: hidden size of generator.

`dis_emb_size`: projection size of discriminator.

`dis_hidden_1_size`: hidden size of the first layer in discriminator.

`dis_hidden_2_size`: hidden size of the second layer in discriminator.

`seg_loss_ratio`: ratio of segment loss.

`penalty_ratio`: ratio of gradient penalty.

`frame_temp`:  temperature of argmax. 

`phn_max_length`: max length of phoneme sequence.

`feat_max_length`: max length of audio feature.

`concat_window`: window size of feature to concatenate.

`sample_var`: variance of sampling process.

`repeat`: number of pair in segment loss.

`batch_size`: batch size.

`epoch`: number of training epoch in supervised model.

`step`: number of training step in unsupervised model.

`print_step`: number of step to print the loss.

`eval_step`: number of step to evalution.

`gen_iter`: iteration of generator.

`dis_iter`: iteration of discriminator.

`gen_lr`: learning rate of generator.

`dis_lr`: learning rate of discriminator.

`sup_lr`: learning rate in supervised model.


## Reference
[Completely Unsupervised Phoneme Recognition By A Generative AdversarialNetwork Harmonized With Iteratively Refined Hidden Markov Models](https://arxiv.org/abs/1904.04100?fbclid=IwAR3QG6ihbKmLz-e4BdOkRG3AaelP5HGkzLkavzRSF6IORN90BkHX1NLkpRo),  Kuan-Yu Chen, Che-Ping Tsai *et.al.*