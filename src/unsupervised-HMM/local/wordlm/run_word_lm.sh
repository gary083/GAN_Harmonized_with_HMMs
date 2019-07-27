#!/bin/bash
stage=0
. ./cmd.sh
. ./path.sh

# you might not want to do this for interactive shells.
set -e

## Given input
prefix=match  # match or nonmatch
phone_map_txt=/groups/public/wfst_decoder/data/timit_new/phones/phones.60-48-39.map.txt
lm_text=/groups/public/wfst_decoder/data/timit_new/text/${prefix}_lm.48
raw_lexicon=./data/lexicon/raw_lexicon.txt
## Parameters
nj=24
n_gram=3

## Output directory
exp=exp/progressive
lexicon=data/lexicon/lexicon.txt
phone_list_txt=data/phone_list.txt
dict=data/local/dict_word
local_lang=data/local/lang_word
lang=data/lang_word
lm_dir=data/${prefix}_word
lm=$lm_dir/$n_gram\gram.lm
lang_test=data/${prefix}_word/lang_test_$n_gram\gram
mfccdir=data/mfcc


if [ $stage -le 0 ]; then
  # Preprocess
  # Format phones.txt and get transcription
  python3 local/trans_lexicon.py $raw_lexicon $phone_map_txt $lexicon
  python3 local/preprocess.py $phone_map_txt $phone_list_txt 
  
  echo "$0: Preparing dict."
  local/prepare_dict.sh $phone_list_txt $dict
  
  echo "$0: Generating lang directory."
  utils/prepare_lang.sh --position_dependent_phones false \
    $dict "<UNK>" $local_lang $lang 
  echo "$0: Creating data." 
  
  cat $lang/words.txt | awk '{print $1 }'  | grep -v "<eps>"  |\
    grep -v "#0" > $lang/vocabs.txt 
  exit 1

  if [ ! -f $lm ]; then
    mkdir -p $lm_dir
    ngram-count -text $lm_text -lm $lm -vocab $lang/vocabs.txt -limit-vocab -order $n_gram
    mkdir -p $lang_test
    local/format_data.sh $lm $lang $lang_test
  fi
fi

