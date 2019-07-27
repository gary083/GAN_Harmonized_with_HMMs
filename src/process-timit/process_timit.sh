#!/bin/bash
ROOT_DIR=$1
TIMIT_DIR=$2

CODE_DIR=$ROOT_DIR/src/process-timit
DATA_DIR=$ROOT_DIR/data
FEATURE_DIR=$DATA_DIR/timit_feature
GAS_DIR=$DATA_DIR/timit_gas


[ -d "$FEATURE_DIR" ] || mkdir -p $FEATURE_DIR

mfcc_dir="$FEATURE_DIR/mfcc"
spec_dir="$FEATURE_DIR/fbank"

[ -d "$mfcc_dir" ] || mkdir -p $mfcc_dir
[ -d "$spec_dir" ] || mkdir -p $spec_dir

echo "Start processing TIMIT:"
# python $CODE_DIR/create_video_list.py $FEATURE_DIR/train_video_list $TIMIT_DIR train  
# python $CODE_DIR/create_video_list.py $FEATURE_DIR/test_video_list  $TIMIT_DIR test 

echo "  a. Extract MFCC..."
# bash $CODE_DIR/mfcc_extractor.sh -i $FEATURE_DIR/train_video_list -o $mfcc_dir/train -t $FEATURE_DIR/tmp
# bash $CODE_DIR/mfcc_extractor.sh -i $FEATURE_DIR/test_video_list  -o $mfcc_dir/test -t $FEATURE_DIR/tmp

echo "  b. Extract FBANK..."
# bash $CODE_DIR/fbank_extractor.sh -i $FEATURE_DIR/train_video_list -o $spec_dir/train -t $FEATURE_DIR/tmp
# bash $CODE_DIR/fbank_extractor.sh -i $FEATURE_DIR/test_video_list  -o $spec_dir/test -t $FEATURE_DIR/tmp

echo "  c. Extract SPEC, PHN, WRD and OTHERS..."
# python $CODE_DIR/gather.py $FEATURE_DIR $TIMIT_DIR train
# python $CODE_DIR/gather.py $FEATURE_DIR $TIMIT_DIR test

# python $CODE_DIR/nor.py $FEATURE_DIR/processed

echo "  d. Split DATASET..."
python $CODE_DIR/split_dataset.py --gas_root $GAS_DIR --timit_root $FEATURE_DIR/processed --save_root $DATA_DIR #--show_bnd_score 
