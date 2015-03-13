#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

OUTDIR=~/Projects/DepthPrediction
TOOLS=$CAFFE_HOME/build/tools

TRAIN_DATA_ROOT=make_mean_hack/
#VAL_DATA_ROOT=/home/vlf/Projects/DepthPrediction/train/NYUv2/test_full_167_v2/

IND_NAME=index.txt
OUT_NAME=mean_hack_lmdb

RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=227
  RESIZE_WIDTH=227
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $TRAIN_DATA_ROOT/$IND_NAME \
    $OUTDIR/$OUT_NAME

#echo "Creating val lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $VAL_DATA_ROOT \
#    $VAL_DATA_ROOT/$IND_NAME \
#    $oOUTDIR/NYUv2_test_$OUT_NAME

echo "Done."
