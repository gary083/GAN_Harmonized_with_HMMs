# GAN_Harmonized_with_HMMs

This is the implementation of [our paper](#Citation). In this paper, we proposed an unsupervised phoneme recogntion system which can achieve 33.1% phoneme error rate on TIMIT.
This method developed a GAN-based model to achieve unsupervised phoneme recognition and we further use a set of HMMs to work in harmony with the GAN.

## How to use

### Dependencies
1. tensorflow 1.13

2. kaldi

3. srilm (can be built with kaldi/tools/install_srilm.sh)

### Data preprocess
- Usage:

1. Modify `path.sh` with your path of Kaldi and srilm.
2. Modify `config` with your code path and timit path.
3. Run `$ bash preprocess.sh`

- This script will extract features and split dataset into train/test set.

- The data which WFST-decoder needed also generate from here.

### Train model
- Usage:

1. Modify the experimental setting in `config`.
2. Modify the GAN-based model's parameter in `src/GAN-based-model/config.yaml`.
2. Run `$ bash run.sh`

- This scipt contains the training flow for GAN-based model and HMM model.

- GAN-based model generated the transcription for training HMM model.

- HMM model refined the phoneme boundaries for training GAN-based model.

## Reference
[Completely Unsupervised Phoneme Recognition By A Generative AdversarialNetwork Harmonized With Iteratively Refined Hidden Markov Models](https://arxiv.org/abs/1904.04100?fbclid=IwAR3QG6ihbKmLz-e4BdOkRG3AaelP5HGkzLkavzRSF6IORN90BkHX1NLkpRo),  Kuan-Yu Chen, Che-Ping Tsai *et.al.*

## Links (Special thanks)
1.  The [WFST decoder](https://github.com/jackyyy0228/WFST-decoder-for-phoneme-posterior) for phoneme classifier<sup>[1](#Reference)</sup> .
2.  The training scripts for [Unsupervised HMM](https://github.com/jackyyy0228/Unsupervised_HMM) <sup>[1](#Reference)</sup> .

## Acknowledgement
**Special thanks to Che-Ping Tsai (jackyyy0228) !**

## Hyperparameters in `config`
`bnd_type` : type of initial phoneme boundaries (orc/uns).

`setting` : matched and nonmatched case in [our paper](#Citation) (match/nonmatch).

`jobs` : number of jobs in parallel (depends on your decive).

