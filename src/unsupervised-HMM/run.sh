#!/bin/bash

stage=0
# . ./cmd.sh
# . ./path.sh

# you might not want to do this for interactive shells.
set -e

## Given input
prefix=$3  # match or nonmatch
exp=$2
phone_map_txt=$1/phones.60-48-39.map.txt
lm_text=$1/timit_for_GAN/text/${prefix}_lm.48
hmm_data=$1/timit_for_HMM/
## Parameters
nj=$4
n_gram=9

## Output directory
phone_list_txt=data/phone_list.txt
dict=data/local/dict
lang=data/lang
lm_dir=data/${prefix}
lm=$lm_dir/$n_gram\gram.lm
lang_test=data/${prefix}/lang_test_$n_gram\gram
mfccdir=$hmm_data/mfcc

if [ $stage -le 0 ]; then
  # Preprocess
  # Format phones.txt and get transcription
  mkdir -p data
  python3 local/preprocess.py $phone_map_txt $phone_list_txt 
  
  echo "$0: Preparing dict."
  local/prepare_dict.sh $phone_list_txt $dict
  
  echo "$0: Generating lang directory."
  utils/prepare_lang.sh --position_dependent_phones false \
    $dict "<UNK>" data/local/lang $lang 
  echo "$0: Creating data." 
  
  cat $lang/words.txt | awk '{print $1 }'  | grep -v "<eps>"  |\
    grep -v "#0" > $lang/vocabs.txt 
  
  if [ ! -f $lm ]; then
    mkdir -p $lm_dir
    ngram-count -text $lm_text -lm $lm -vocab $lang/vocabs.txt -limit-vocab -order $n_gram
    mkdir -p $lang_test
    local/format_data.sh $lm $lang $lang_test
  fi
fi

if [ $stage -le 1 ]; then
  for part in train test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj $hmm_data/$part $hmm_data/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh $hmm_data/$part $hmm_data/make_mfcc/$part $mfccdir
  done
  cp -r $hmm_data/train/*.scp $hmm_data/train_correct/
fi

if [ $stage -le 2 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
                      $hmm_data/train data/lang $exp/mono
  utils/mkgraph.sh $lang_test \
                   $exp/mono $exp/mono/graph
  #decode using the monophone model
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $exp/mono/graph \
                    $hmm_data/test $exp/mono/decode_test
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $exp/mono/graph \
                  $hmm_data/train $exp/mono/decode_train
fi

if [ $stage -le 3 ]; then
  new_data=$hmm_data/train_mono
  prev_gmm=$exp/mono
  ali=$exp/mono_ali
  gmm=$exp/tri1
  cp -r $hmm_data/train $new_data
  cat $prev_gmm/decode_train/scoring_kaldi/penalty_1.0/1.txt | sort > $new_data/text

  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
                   $new_data data/lang $prev_gmm $ali

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2500 15000 $new_data data/lang $ali $gmm

  utils/mkgraph.sh $lang_test \
                   $gmm $gmm/graph
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $gmm/graph \
                    $hmm_data/test $gmm/decode_test
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $gmm/graph \
                  $hmm_data/train_correct $gmm/decode_train
  rm -r $new_data
fi

if [ $stage -le 4 ]; then
  new_data=$hmm_data/train_tri1
  prev_gmm=$exp/tri1
  ali=$exp/tri1_ali
  gmm=$exp/tri2

  cp -r $hmm_data/train $new_data
  cat $prev_gmm/decode_train/scoring_kaldi/penalty_1.0/1.txt | sort > $new_data/text

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                   $new_data data/lang $prev_gmm $ali

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                       $new_data data/lang $ali $gmm

  utils/mkgraph.sh $lang_test \
                   $gmm $gmm/graph
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $gmm/graph \
                  $hmm_data/test $gmm/decode_test
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $gmm/graph \
                  $hmm_data/train_correct $gmm/decode_train
  rm -r $new_data
fi

if [ $stage -le 5 ]; then
  new_data=$hmm_data/train_tri3
  prev_gmm=$exp/tri2
  ali=$exp/tri2_ali
  gmm=$exp/tri3

  cp -r $hmm_data/train $new_data
  cat $prev_gmm/decode_train/scoring_kaldi/penalty_1.0/1.txt | sort > $new_data/text

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                   $new_data data/lang $prev_gmm $ali

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                       $new_data data/lang $ali $gmm

  utils/mkgraph.sh $lang_test \
                   $gmm $gmm/graph
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $gmm/graph \
                  $hmm_data/test $gmm/decode_test
  steps/decode.sh --nj $nj --cmd "$decode_cmd" $gmm/graph \
                  $hmm_data/train_correct $gmm/decode_train
  rm -r $new_data
fi

if [ $stage -le 6 ]; then
  for dir in mono tri1 tri2 tri3; do
    bash local/score.sh $hmm_data/test $lang_test $exp/$dir/decode_test
    cat $exp/$dir/decode_test/scoring_kaldi/penalty_1.0/1.txt | sort > $exp/${dir}_test_output.txt
  done
fi