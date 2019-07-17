# GAN_Harmonized_with_HMMs

This is the implementation of [our paper](#Citation).

## How to use

### Dependencies
1. tensorflow 1.13

2. kaldi

3. srilm (can be built with kaldi/tools/install_srilm.sh)

### Data preprocess
1. Modify the root and TIMIT directory in src/preprocess.sh
2. Run 
```
bash preprocess.sh
```

## Reference
[Completely Unsupervised Phoneme Recognition By A Generative AdversarialNetwork Harmonized With Iteratively Refined Hidden Markov Models](https://arxiv.org/abs/1904.04100?fbclid=IwAR3QG6ihbKmLz-e4BdOkRG3AaelP5HGkzLkavzRSF6IORN90BkHX1NLkpRo),  Kuan-Yu Chen, Che-Ping Tsai *et.al.*

## Links 
1.  The [WFST decoder](https://github.com/jackyyy0228/WFST-decoder-for-phoneme-posterior) for phoneme classifier<sup>[2](#Reference)</sup> .
2.  
