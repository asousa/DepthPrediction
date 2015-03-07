# Creates dataset directories (jpegs with an index file)

import numpy as np
import sys
import matplotlib.pyplot as plt

from image_handling import load_dataset
from image_handling import segment_image

# Generate full datasets, using splits file: 
import scipy.io

# Load splits
splits_path = '/home/vlf/Projects/DepthPrediction/train/NYUv2/splits.mat'
splits = scipy.io.loadmat(splits_path)
train_inds = splits['trainNdxs']
test_inds = splits['testNdxs']

from image_handling import create_segments_dataset
from image_handling import create_segments_directory

min_depth = 0.7;
max_depth = 10;
depth_bins = 16;

window_size = 83; # 167x167~NYUv2 paper # 227x227, imagenet standard
n_superpixels = 200;

# Generate training set

input_file = 'train/nyu_depth_v2_labeled.mat'
img_filepath_train = '/home/vlf/Projects/DepthPrediction/train/NYUv2/train_full_167'
img_filepath_test = '/home/vlf/Projects/DepthPrediction/train/NYUv2/test_full_167'

train_slices = np.array_split(train_inds - 1,10)
test_slices = np.array_split(test_inds - 1, 10)

# print 'Building training set...'
# for s in train_slices:
#    images = tuple(s)
#    data_file = create_segments_directory(input_filename=input_file, image_output_filepath=img_filepath_train,
#                    no_superpixels=n_superpixels, x_window_size=window_size, y_window_size=window_size, images=images,
#                    depth_bins=depth_bins,depth_min=min_depth, depth_max=max_depth)

print 'Building validation set...'
for s in test_slices:
    images = tuple(s)
    data_file = create_segments_directory(input_filename=input_file, image_output_filepath=img_filepath_test,
                    no_superpixels=n_superpixels, x_window_size=113, y_window_size=113, images=images,
                    depth_bins=depth_bins,depth_min=min_depth, depth_max=max_depth)

    
