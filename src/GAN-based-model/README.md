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
`mode`:
`model_type`:
`cuda_id`:
`bnd_type`:
`setting`:
`iteration`:
`data_dir`:
`save_dir`:
`config`:



## Hyperparameters in `config.yaml`

## Reference
[Completely Unsupervised Phoneme Recognition By A Generative AdversarialNetwork Harmonized With Iteratively Refined Hidden Markov Models](https://arxiv.org/abs/1904.04100?fbclid=IwAR3QG6ihbKmLz-e4BdOkRG3AaelP5HGkzLkavzRSF6IORN90BkHX1NLkpRo),  Kuan-Yu Chen, Che-Ping Tsai *et.al.*