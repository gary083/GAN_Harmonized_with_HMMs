#!/bin/bash
# . path.sh

word_symtab=data/lang/words.txt
phone_symtab=data/lang/phones.txt
exp=$1
mdl=$exp/mono/final.mdl
lmwt=7
penalty=1
nj=$2
typ=mono

. ./utils/parse_options.sh || exit 1;

decode_dir=$exp/mono/decode_train

pdf_symtab=data/lang/pdfs_${typ}.txt

if [ ! -f $pdf_symtab ] ; then
  python3 local/phones2pdf.py --typ $typ $phone_symtab $pdf_symtab
fi

for i in `seq $nj`; do
  lattice-scale --inv-acoustic-scale=$lmwt "ark:gunzip -c $decode_dir/lat.$i.gz|" ark:- | \
    lattice-add-penalty --word-ins-penalty=$penalty ark:- ark:- | \
    lattice-best-path --word-symbol-table=$symtab ark:- "ark,t:|utils/int2sym.pl -f 2- $word_symtab  > $decode_dir/ali_output.$i.txt" ark:- | \
    ali-to-pdf $mdl ark:- "ark,t:|utils/int2sym.pl -f 2- $pdf_symtab > $decode_dir/phones_ali.$i.txt" & 
done
wait

cat $decode_dir/phones_ali.*.txt | sort > $exp/phones_ali.txt
rm $decode_dir/ali_output.*.txt $decode_dir/phones_ali.*.txt

echo "The path of alignment file is $exp/phones_ali.txt"
echo "Done."
