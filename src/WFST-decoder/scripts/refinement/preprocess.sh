#!/bin/bash

. ./cmd.sh
. ./path.sh
##prepare lang

stage=0
## Given input
prefix=match  # match or nonmatch
phone_map_txt=data/timit_new/phones/phones.60-48-39.map.txt
lm_text=data/timit_new/text/${prefix}_lm.48

## Hyperparameters
self_loop_prob=0.5
n_gram=9
. ./utils/parse_options.sh

## Output directory
monophone_list_txt=data/refinement/monophone_list.txt
dict=data/refinement/local/dict
lang=data/refinement/lang
lm_dir=data/${prefix}
lm=$lm_dir/$n_gram\gram.lm
lang_test=data/refinement/${prefix}/lang_test_$n_gram\gram
treedir=data/refinement/${prefix}/tree_sp$self_loop_prob  # it's actually just a trivial tree (no tree building)

if [ $stage -le 0 ]; then
  # Preprocess
  # Format phones.txt and get transcription
  mkdir -p data/refinement $dict
  python3 scripts/refinement/get_monophone_list.py $phone_map_txt $monophone_list_txt
  
  echo "$0: Preparing dict."
  python3 scripts/refinement/prepare_dict.py $monophone_list_txt $dict
  echo "$0: Generating lang directory."
  
  utils/prepare_lang.sh --position_dependent_phones false \
    $dict "<UNK>" data/refinement/local/lang $lang 
  
  # customize fst
  scripts/refinement/generate_context_fst.sh $monophone_list_txt $self_loop_prob $lang/phones.txt $lang/b.fst
  
  fsttablecompose $lang/b.fst $lang/L_disambig.fst  | \
     fstdeterminizestar --use-log=true | \
     fstminimizeencoded  | fstarcsort --sort_type=olabel > $lang/bL_disambig.fst
  mv $lang/L_disambig.fst $lang/L_disambig_ori.fst
  mv $lang/bL_disambig.fst $lang/L_disambig.fst

  echo "$0: Creating data." 
fi

if [ $stage -le 1 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  mkdir -p $treedir
  
  python3 scripts/refinement/gen_topo.py $lang/phones.txt \
    --self_loop_prob $self_loop_prob > $treedir/topo
  
  ## Initiialize a tree and a transition model
  echo "$0: Initializing mono phone system."
  # feat dim does not matter here. Just set it to 10
  run.pl $treedir/log/init_mono_mdl_tree.log \
       gmm-init-mono  $treedir/topo 10 \
       $treedir/0.mdl $treedir/tree || exit 1;
  copy-transition-model $treedir/0.mdl $treedir/0.trans_mdl
  ln -s 0.mdl $treedir/final.mdl  # for consistency with scripts which require a final.mdl
fi

if [ $stage -le 2 ]; then
  echo "$0: Training phone lm and transform it to fst."
  #remove disambig symbols
  cat $lang/words.txt | awk '{print $1 }'  | grep -v "<eps>"  |\
    grep -v "#0" > $lang/vocabs.txt 
  
  mkdir -p $lm_dir
  if [ ! -f $lm ]; then
    ngram-count -text $lm_text -lm $lm -vocab $lang/vocabs.txt -limit-vocab -order $n_gram
  fi
  if [ ! -d $lang_test ]; then
    scripts/format_data.sh $lm $lang $lang_test
  fi
  
  echo "$0:Compiling graph for decoding."
  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_test \
    $treedir $treedir/graph_$n_gram\gram  || exit 1;
fi

