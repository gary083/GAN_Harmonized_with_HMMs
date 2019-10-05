#!/bin/bash
#. path.sh
# dir=$1

word_symtab=data/lang/words.txt
phone_symtab=data/lang/phones.txt
exp=$1
mdl=$exp/mono/final.mdl
lmwt=7
penalty=1
nj=$2
typ=mono

dir=$exp/mono

gunzip -c  $dir/ali.*.gz | ali-to-phones --per-frame $mdl ark:- ark,t:- | utils/int2sym.pl  -f 2- data/lang/phones.txt  | sort > $exp/phones_ali.txt

echo "The path of alignment file is $exp/phones_ali.txt"
echo "Done."

