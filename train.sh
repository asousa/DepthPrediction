#!/bin/bash
# Pre-train a regressor
echo "Pretraining..."

# starting point

#INITIAL_WEIGHTS="models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
#SOLVER_PROTO="models/finetune_flickr_style/solver.prototxt"

INITIAL_WEIGHTS="models/placesCNN/places205CNN_iter_300000.caffemodel"
#INITIAL_WEIGHTS="models/unary_depth_regressor/snapshots/udr_softmax_lmdb_snapshot_iter_7000.caffemodel" # resume from here
#SOLVER_PROTO="models/unary_depth_regressor/solver_udr_softmax_lmdb.prototxt"	
#SOLVER_PROTO="models/unary_depth_regressor/solver_udr_euclidean_lmdb.prototxt"

# Softmax
#SOLVER_PROTO="models/unary_depth_regressor/solver_euclidean_dir.prototxt"
#SOLVER_PROTO="models/unary_depth_regressor/solver_euclidean_no_softmax.prototxt"
#SOLVER_PROTO="models/udr_scratch/solver_softmax_32bins.prototxt"
SOLVER_PROTO="models/unary_depth_regressor/solver_softmax_dir.prototxt"
# Euclidean
#SOLVER_PROTO="models/unary_depth_regressor/solver_euclidean_dir.prototxt"
#SNAPSHOT_PATH="models/unary_depth_regressor/snapshots_euclidean/udr_euclidean_dir_v3_16_norm_realmean_iter_10000.solverstate"
#INITIAL_WEIGHTS='models/unary_depth_regressor/snapshots_euclidean/udr_euclidean_d16_superpixel_avg_hardcoded_output_layer_iter_20000.caffemodel'

# Use this to start:
caffe train -solver $SOLVER_PROTO -weights $INITIAL_WEIGHTS -gpu 0

# Use this to resume
#caffe train -solver $SOLVER_PROTO -snapshot $SNAPSHOT_PATH -gpu 0

# Use this to run from scratch (probably terrible idea)
#caffe train -solver $SOLVER_PROTO
