#!/bin/bash
. path.sh
dir=$1

gunzip -c  $dir/ali.*.gz | ali-to-phones --per-frame $dir/final.mdl ark:- ark,t:- | utils/int2sym.pl  -f 2- data/lang/phones.txt  | sort > $dir/ali_output.txt

python3 local/eval_bound.py $dir/ali_output.txt
