#!/usr/bin/env sh
# Compute the mean image from a training database


TRAIN_DB_PATH=mean_hack_lmdb
OUTPUT_PATH=mean.binaryproto

compute_image_mean $TRAIN_DB_PATH \
  $OUTPUT_PATH

echo "Done."
