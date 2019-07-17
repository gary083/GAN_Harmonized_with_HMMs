#!/bin/bash

usage() { echo "Usage: $0 -i <video_file_list> -o <feat_dir> -t <tmp_folder>" 1>&2; exit 1; }

while getopts ":i:o:t:" arg; do
    case "${arg}" in
        i)
            video_file_list=${OPTARG}
            ;;
        o)
            feat_dir=${OPTARG}
            ;;
        t)
            tmp_folder=${OPTARG}
            ;;   
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z "${video_file_list}" ] || [ -z "${feat_dir}" ]; then
    usage
fi

if [ -z "${tmp_folder}" ]; then
    tmp_folder="./tmp/"
fi

log="$feat_dir/fbank_log"
# log="./fbank_log"
# check if feat directory exists
[ -d "$feat_dir" ] || mkdir $feat_dir
[ -d "$tmp_folder"  ] || mkdir $tmp_folder
list_name=$video_file_list
doc_len=`cat $list_name |wc -l`

function main_function {
    count=0
    while read LINE
    do
        video_name=$(echo $(basename $LINE) | cut -d '.' -f 1)
        count=$(($count+1))
        echo "Processing " $video_name " " $count"/"$doc_len"."
        # extract fbank feats
        echo "Extract fbank feats..."
        echo "$video_name ${LINE}.wav" > $tmp_folder/tmp.scp
        compute-fbank-feats --sample-frequency=16000 --channel=0 scp:$tmp_folder/tmp.scp ark,t,scp:$tmp_folder/tmp.$video_name.13.ark,$tmp_folder/tmp.$video_name.13.scp
        wait $!
        add-deltas scp:$tmp_folder/tmp.$video_name.13.scp ark,t,scp:$tmp_folder/tmp.$video_name.delta.ark,$tmp_folder/tmp.$video_name.delta.scp
        wait $!
        compute-cmvn-stats --binary=false scp:$tmp_folder/tmp.$video_name.delta.scp ark,t:$tmp_folder/tmp.$video_name.cmvn.ark
        wait $!
        apply-cmvn --norm-vars=false ark:$tmp_folder/tmp.$video_name.cmvn.ark scp:$tmp_folder/tmp.$video_name.delta.scp ark,t,scp:$feat_dir/$video_name.fbank.cmvn.ark,$feat_dir/$video_name.fbank.cmvn.scp
        wait $!
        rm $tmp_folder/tmp*
    done < $list_name
}
# main_function 2>&1 | tee -a $log
main_function >> $log 2>&1 
rm -r $tmp_folder
