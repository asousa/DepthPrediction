#!/usr/bin/env sh
# Compute the mean image from a training database


TRAIN_DB_PATH=train/NYUv2/NYUv2_train_full_167_resize_d16_lmdb
OUTPUT_PATH=train/NYUv2/NYUv2_train_mean_full_167_resize_d16.binaryproto

compute_image_mean $TRAIN_DB_PATH \
  $OUTPUT_PATH

echo "Done."
