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
2. Training Phase 
```
python3 main.py --mode train --cuda_id 0 
				--bnd_type  [boundaries-type]
				--setting   [real-data-type] 
				--iteration [iteration-of-HMM-hamonization]
				--data_dir  [path-to-data-directory]
				--save_dir  [path-to-save-directory]
				--config    [path-to-config]

```


## Hyperparameters in `config.yaml`

## Reference
[Completely Unsupervised Phoneme Recognition By A Generative AdversarialNetwork Harmonized With Iteratively Refined Hidden Markov Models](https://arxiv.org/abs/1904.04100?fbclid=IwAR3QG6ihbKmLz-e4BdOkRG3AaelP5HGkzLkavzRSF6IORN90BkHX1NLkpRo),  Kuan-Yu Chen, Che-Ping Tsai *et.al.*