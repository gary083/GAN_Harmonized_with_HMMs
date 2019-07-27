#!/bin/bash

# Copyright 2013  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Script for system combination using minimum Bayes risk decoding.
# This calls lattice-combine to create a union of lattices that have been 
# normalized by removing the total forward cost from them. The resulting lattice
# is used as input to lattice-mbr-decode. This should not be put in steps/ or 
# utils/ since the scores on the combined lattice must not be scaled.

# begin configuration section.
cmd=run.pl
min_lmwt=1
max_lmwt=10
lat_weights=
stage=0
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <data-dir> <graph-dir|lang-dir> <decode-dir1> <decode-dir2> <out-dir>
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --min-lmwt INT                  # minumum LM-weight for lattice rescoring 
  --max-lmwt INT                  # maximum LM-weight for lattice rescoring
  --lat-weights STR               # colon-separated string of lattice weights
";

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 5 ]; then
  printf "$help_message\n";
  exit 1;
fi

data=$1
graphdir=$2
decode_dir1=$3  # read the remaining arguments into an array
decode_dir2=$4
odir=$5
nj=`cat $decode_dir1/num_jobs`

# Map reference to 39 phone classes, the silence is optional (.):
mkdir -p $odir
if [ $stage -le 0 ]; then
  for x in `seq $nj` ; do
    lattice-combine "ark:gunzip -c $decode_dir1/lat.$x.gz|" "ark:gunzip -c $decode_dir2/lat.$x.gz|"  ark:- | gzip  > $odir/lat.$x.gz &
  done
  wait
fi

[ ! -x local/score.sh ] && \
  echo "$0: Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd"  $data $graphdir $odir ||
  { echo "$0: Error: scoring failed. (ignore by '--skip-scoring true')"; exit 1; }

exit 0
