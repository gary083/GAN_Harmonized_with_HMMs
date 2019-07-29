#!/bin/bash

phone_list=$1
dict=$2

mkdir -p $dict

touch $dict/extra_questions.txt
echo "sil" >  $dict/optional_silence.txt
echo "sil" > $dict/silence_phones.txt
echo "spn" >> $dict/silence_phones.txt
cut -d' ' -f2- $phone_list | grep -v spn - | grep -v sil - > $dict/nonsilence_phones.txt
cut -d' ' -f2- $phone_list | awk '{print $0 " " $0}' - > $dict/lexicon.txt
echo "<UNK> spn" >> $dict/lexicon.txt

echo "Done:$0"

