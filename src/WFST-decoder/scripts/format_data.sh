#/bin/bash
. path.sh


if [ -f ./path.sh ]; then . ./path.sh; fi

arpa_lm=$1
lang=$2
lang_test=$3
. ./utils/parse_options.sh

[ ! -f $arpa_lm ] && echo No such file $arpa_lm && exit 1;


rm -r $lang_test
cp -r $lang $lang_test

echo $arpa_lm
cat $arpa_lm | arpa2fst --disambig-symbol=#0 \
  --read-symbol-table=$lang_test/words.txt -  | fstarcsort --sort_type=olabel > $lang_test/G.fst
#cat $arpa_lm | arpa2fst - |  fstprint |\
#   fstcompile --isymbols=$lang_test/words.txt --osymbols=$lang_test/words.txt \
#   --keep_isymbols=false --keep_osymbols=false | fstrmepsilon | fstarcsort --sort_type=olabel > $lang_test/G.fst

echo  "Checking how stochastic G is (the first of these numbers should be small):"
fstisstochastic $lang_test/G.fst

## Check lexicon.
## just have a look and make sure it seems sane.
echo "First few lines of lexicon FST:"
fstprint   --isymbols=$lang/phones.txt --osymbols=$lang/words.txt $lang/L.fst  | head

echo Performing further checks

# Checking that G.fst is determinizable.
fstdeterminize $lang_test/G.fst /dev/null || echo Error determinizing G.

# Checking that L_disambig.fst is determinizable.
fstdeterminize $lang_test/L_disambig.fst /dev/null || echo Error determinizing L.

# Checking that disambiguated lexicon times G is determinizable
# Note: we do this with fstdeterminizestar not fstdeterminize, as
# fstdeterminize was taking forever (presumbaly relates to a bug
# in this version of OpenFst that makes determinization slow for
# some case).
fsttablecompose $lang_test/L_disambig.fst $lang_test/G.fst | \
   fstdeterminizestar >/dev/null || echo Error

# Checking that LG is stochastic:
fsttablecompose $lang/L_disambig.fst $lang_test/G.fst | \
   fstisstochastic || echo LG is not stochastic

echo format_data succeeded.

