#/bin/bash
monophone_txt=$1
self_loop_prob=$2
phone_txt=$3
out_fst=$4
text_fst=`dirname $out_fst`
text_fst=$text_fst/text.fst

. path.sh

python3 scripts/refinement/generate_context_fst.py $monophone_txt $phone_txt $self_loop_prob > $text_fst

fstcompile --keep_isymbols=false --keep_osymbols=false $text_fst | fstarcsort --sort_type=olabel  > $out_fst

