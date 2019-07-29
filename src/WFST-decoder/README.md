# WFST decoder for phoneme posterior

This is a WFST decoder for phoneme posterior built on kaldi.

Feel free to use/modify them, any bug report or improvement suggestion will be appreciated. 

<!--If you find this project helpful for your research, please do consider to cite our paper, thanks! -->

##  Composition of WFST

The WFST is composed of HCLG. 

**H** is an 2 state HMM with probability of 0.95 for self-loop and probabilty of 0.05 to transit to final state.

**C** is an one-to-one mapping function of phoneme, which is built from a full-unigram tree.

**L** is an one-to-one mapping of all phoneme.

**G** is an FST built from a 9-gram phone LM.

## How to use

### Dependencies

- kaldi 

- srilm (can be built with kaldi/tools/install_srilm.sh)

### Path

- Modify path.sh with your path of kaldi and srilm.

### Preprocess

- Format and make lang directory for data preparation of kaldi. (Should be modified with different kinds of input format)

- Train n-gram phone LM and generate HMM topology with specified self-loop prob.

```
$ bash scripts/preprocess.sh --self_loop_prob 0.95 --n_gram 9
```

### Decode

- Decode with given phoneme posteriors and graph compiled in preprocess.sh and automatically compute PER corresponding to correct transcription.

- The order of phoneme may be different (should be transformed manually) , since there is a specified order by kaldi (can be checked in data/lang/phones.txt).


```
$  python scripts/decode.py 
```

### Alignment

- Get phoneme alignment from lattices in decoding directory.

```
$  bash scripts/lat_lat_to_phones.sh $decode_dir
```

### Joint decode with segmentation probabilities

-  Consider segmentation probabillity in WFST decoding, e.g. GAS probability in [1].

```
$  bash scripts/refinement/preprocess.sh --self_loop_prob 0.95 --n_gram 9
$  python scripts/refinement/decode.py 
```

## Reference

1.[Unsupervised Speech Recognition via Segmental Empirical Output Distribution Matching.](https://arxiv.org/abs/1812.09323), Chih-Kuan Yeh*et al.*

2.[Completely Unsupervised Phoneme Recognition By A Generative AdversarialNetwork Harmonized With Iteratively Refined Hidden Markov Models](https://arxiv.org/abs/1904.04100?fbclid=IwAR3QG6ihbKmLz-e4BdOkRG3AaelP5HGkzLkavzRSF6IORN90BkHX1NLkpRo),  Kuan-Yu Chen, Che-Ping Tsai *et.al.*

## Useful Links

1.  The training scripts for [Unsupervised HMM](https://github.com/jackyyy0228/Unsupervised_HMM) <sup>[2](#Reference)</sup>

