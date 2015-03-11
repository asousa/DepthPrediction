import h5py
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import os
import scipy
import matplotlib.pyplot as plt

def load_dataset(filename=None):
	""" 
	Load in the hdf5 dataset from the specified filename with the labels
	'images' and 'depths'.  Return these two hdf5 objects as a tuple.
	"""
	nyu_set = h5py.File(filename, 'r')
	images = nyu_set['images']
	depths = nyu_set['depths']
	return [images, depths]

def log_pixelate_values(array, min_val, max_val, bins):
	"""
	Given an array or an hdf5 object, log_pixelate_values takes the values along
	the array and bins them into integer values representing bins that are log
	spaced from the min to the max value. This should work on any dimension of
	array.
	"""
	print bins
	cuts = np.logspace(np.log(min_val), np.log(max_val), num=(bins+1), base=np.e)
	print cuts.shape
	array_vals = np.array(array[:])
	val = np.reshape(np.digitize(array_vals.flatten(), cuts), array.shape)
	return val.astype(int)


def segment_image(image=None, no_segments=500):
	"""
	Break the images into no_segments parts using the SLIC algorithm.  The
	algorithm does not guarantee no_segments will be found.
	"""
	if len(image.shape) != 3:
		raise ValueError('Dimension of image is not 3')
	if image.shape[0] != 3:
		raise ValueError('First dimension of image is not 3')
	mask = slic(np.transpose(image, (1, 2, 0)), n_segments=no_segments, compactness=15, sigma=1)
	return mask

def calculate_sp_centroids(mask=None):
	"""
	Generates an average position of each superpixel in x and y.  Return a
	2 x n array with x in (0, :) and y in (1, :)
	"""
	pixel_ids = np.unique(mask)

	# Thought I could vectorize this an make it faster... nope
	#
	# pixel_com = np.tile(pixel_ids, [mask.shape[0], mask.shape[1], 1])
	# mask_com = np.tile(mask, [len(pixel_ids), 1, 1]).transpose(1, 2, 0)
	# com_mat = mask_com == pixel_com

	# pixel_counts = com_mat.sum(axis=(0,1))

	# weight_x = np.tile(range(0, mask.shape[0]), [len(pixel_ids), 1]).transpose()
	# weight_y = np.tile(range(0, mask.shape[1]), [len(pixel_ids), 1]).transpose()

	# centroids = np.vstack(((com_mat.sum(axis=1) * weight_x).sum(0) / pixel_counts, 
	# 	                   (com_mat.sum(axis=0) * weight_y).sum(0) / pixel_counts))

	centroids = np.zeros((2, len(pixel_ids)))
	weight_x = np.array(range(0, mask.shape[0]))
	weight_y = np.array(range(0, mask.shape[1]))

	for pixel in pixel_ids:
		total = np.sum(mask == pixel, axis=None, dtype=float)
		centroids[0, pixel] = np.sum(weight_x * np.sum(mask == pixel, axis=1, dtype=float) / total)
		centroids[1, pixel] = np.sum(weight_y * np.sum(mask == pixel, axis=0, dtype=float) / total)

	return centroids

def gather_regions(image=None, centroids=None, x_window_size=10, y_window_size=10):
	"""
	Generate an np.ndarray containing a box around all of the superpixels
	within the image.  The box is 1 + 2 * window_size in the x and y
	directions.  Images whos boxes would hang over the edge of the image are
	moved up to the boundary edge, so that the centroid sits off-center closer
	to the edge of the image.
	"""
	regions = np.zeros((centroids.shape[1], 3, 2 * x_window_size + 1,  2 * y_window_size + 1))
	center_pixels = np.array(centroids, dtype=int)
	for pixel in range(0, center_pixels.shape[1]):
		x = center_pixels[0, pixel]
		y = center_pixels[1, pixel]
		x_left = x - x_window_size - 1
		x_right = x + x_window_size
		y_left = y - y_window_size - 1
		y_right = y + y_window_size

		if x_left < 0:
			x_left = 0
			x_right = 2 * x_window_size + 1
		elif x_right >= image.shape[1]:
			x_left = image.shape[1] - 2 * x_window_size - 2
			x_right = image.shape[1] - 1

		if y_left < 0:
			y_left = 0
			y_right = 2 * y_window_size + 1
		elif y_right >= image.shape[2]:
			y_left = image.shape[2] - 2 * y_window_size - 2
			y_right = image.shape[2] - 1

		regions[pixel, ...] = image[:, x_left:x_right, y_left:y_right]
	
	return regions

def load_dataset_segments(
	filename=None,
	no_superpixels=500,
	images=None,
	x_window_size=10,
	y_window_size=10):
	"""
	Combines all of the above to load images segments and their associated
	depths from a dataset and and return them as a tuple of ndarrays.
	"""
	if images == None:
		images = range(0, images.shape[0])
	if type(images) is not tuple:
		images = range(0, images)
	
	[image_set, depths] = load_dataset(filename)
	no_segments = no_superpixels * len(images)
	segment_depths = np.zeros((no_segments, 1))
	image_segments = np.ndarray((no_segments,
								 image_set.shape[1],
								 2 * x_window_size + 1,
								 2 * y_window_size + 1))
	masks = np.zeros((len(images), depths.shape[1], depths.shape[2]))

	current_image = 0
	current_segment = 0
	for image_idx in images:
		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
		masks[current_image, ...] = mask
		centroids = calculate_sp_centroids(mask)
		center_pixels = np.array(centroids, dtype=int)

		end_index = current_segment + centroids.shape[1]
		if end_index >= image_segments.shape[0]:
			image_segments.resize((end_index + 1,) + image_segments.shape[1:])
		if end_index >= segment_depths.shape[0]:
			segment_depths.resize((end_index + 1,) + segment_depths.shape[1:])

		image_segments[current_segment:(current_segment+centroids.shape[1]), ...] = \
			gather_regions(image, centroids, x_window_size=x_window_size, y_window_size=y_window_size)
		for depth_idx in range(0, centroids.shape[1]):
			segment_depths[current_segment + depth_idx] = \
				depths[image_idx, center_pixels[0, depth_idx], center_pixels[1 , depth_idx]]
		current_segment = current_segment + centroids.shape[1]
		current_image = current_image + 1

	return image_segments[0:current_segment, ...], segment_depths[0:current_segment, ...], masks



def create_segments_dataset(
	input_filename=None,
	output_filename=None,
	no_superpixels=500,
	x_window_size=10,
	y_window_size=10,
	images=None,
	image_output_filepath=None,
	depth_bins=None, depth_min = None, depth_max=None):
	"""
	Combines all of the above to load images segments and their associated
	depths from a dataset and and return them as a tuple of ndarrays.

	-To output a directory of images w/ index file, provide image_output_filepath.
	-To quantize delivered depths into bins, provide depth_bins, depth_min, depth_max
	"""

	if images == None:
		images = range(0, image_set.shape[0])
	if type(images) is not tuple:
		images = range(0, images)
	
	[image_set, depths] = load_dataset(input_filename)
	no_segments = no_superpixels * len(images)

	# Check whether to output individual images
	indiv_output = (image_output_filepath is not None)
	if indiv_output:
		if not os.path.exists(image_output_filepath):
			os.makedirs(image_output_filepath)
		out_log = open(image_output_filepath + '/index.txt','w+')

	# Check if exporting an hdf5 file
	hdf5_output = (output_filename is not None)
	if hdf5_output:
		output_file = h5py.File(output_filename, 'w')
		image_segments = output_file.create_dataset("data",
			(no_segments, image_set.shape[1], 2 * x_window_size + 1, 2 * y_window_size + 1),
			chunks=(1, image_set.shape[1], 2 * x_window_size + 1, 2 * y_window_size + 1))


		segment_depths = output_file.create_dataset("label", (no_segments, 1), chunks=True)
		segment_image_index = output_file.create_dataset("image", (no_segments, 1), chunks=True)
		segment_superpixel_index = output_file.create_dataset("pixel", (no_segments, 1), chunks=True)
	else:
		segment_image_index = []
		segment_superpixel_index = []
		image_segments = []

	current_segment = 0
	for image_idx in images:
		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
		centroids = calculate_sp_centroids(mask)
		center_pixels = np.array(centroids, dtype=int)

		# Resize the arrays if they ended up being too small.
		# Will probably only be called on the last image if at all.
		end_index = current_segment+centroids.shape[1]

		if (hdf5_output):
			if end_index >= image_segments.shape[0]:
				image_segments.resize((end_index + 1,) + image_segments.shape[1:])
				segment_depths.resize((end_index + 1,) + segment_depths.shape[1:])
				segment_image_index.resize((end_index + 1,) + segment_image_index.shape[1:])
				segment_superpixel_index.resize((end_index + 1,) + segment_superpixel_index.shape[1:])

		# Pull out sections around the centroid of the superpixel
		image_segments[current_segment:end_index, ...] = \
				gather_regions(image, centroids,
						x_window_size=x_window_size,
						y_window_size=y_window_size)

		# Pull out the appropriate depth images.
		for depth_idx in range(0, centroids.shape[1]):
			segment_depths[current_segment + depth_idx] = \
					depths[image_idx,
					       center_pixels[0, depth_idx],
						   center_pixels[1, depth_idx]]

 		# Convert depths to quantized logspace:
 		if (depth_bins is not None):
 			print 'quantizing depths'
 			segment_depths[current_segment:current_segment+centroids.shape[1]] = \
 			log_pixelate_values(segment_depths[current_segment:current_segment+centroids.shape[1]],
 				depth_bins, depth_min, depth_max)

 		#print image_segments[current_segment:end_index, ...].shape
 		#print end_index-current_segment
 		if indiv_output:
			for i in range(current_segment,end_index):
				name = image_output_filepath + '/' + str(image_idx) + '_' + str(i) + '.jpg'
				# write image
				#print image_segments[i, ...].shape
				#plt.imshow(image_segments[i, ...])
				scipy.misc.imsave(name,np.transpose(image_segments[i, ...],(0,2,1)))
				# append to log
				#print segment_depths[i]
				out_log.write(name + ' ' + str(int(segment_depths[i][0])) + '\n')


		current_segment = current_segment + centroids.shape[1]
	# If the number of superpixels was smaller than we expected, resize the
	# arrays before returning them
	if current_segment != image_segments.shape[0]:
		image_segments.resize((current_segment,) + image_segments.shape[1:])
		segment_depths.resize((current_segment,)  + segment_depths.shape[1:])
		segment_image_index.resize((current_segment,) + segment_image_index.shape[1:])
		segment_superpixel_index.resize((current_segment,) + segment_superpixel_index.shape[1:])
	return output_file


def apply_depths(segment_depths, mask):
	depth_image = np.zeros(mask.shape, dtype=segment_depths.dtype)
	for depth_index in range(0, len(segment_depths)):
		depth_image += segment_depths[depth_index] * (mask == depth_index)
	return depth_image

def create_segments_directory(
	input_filename=None,
	image_output_filepath=None,
	no_superpixels=200,
	x_window_size=10,
	y_window_size=10,
	images=None,
	depth_bins=None, depth_min = None, depth_max=None,
	output_images=True, index_name='index.txt'):
	"""
	outputs a directory of image segments, with index file.
	"""
	#print 'loop depth_bins = ', depth_bins
	# Select which images to work with
	if images == None:
		images = range(0, image_set.shape[0])
	if type(images) is not tuple:
		images = range(0, images)
	
	[image_set, depths] = load_dataset(input_filename)
	no_segments = no_superpixels * len(images)

	# Create output directory
	if not os.path.exists(image_output_filepath):
		os.makedirs(image_output_filepath)
	out_log = open(image_output_filepath + '/' + index_name,'a')

	image_segments = np.ndarray([no_segments,3,2*x_window_size+1, 2*y_window_size+1])
	segment_depths = np.ndarray(no_segments)
	current_segment = 0
	for image_idx in images:

		print 'processing image', image_idx

		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
		centroids = calculate_sp_centroids(mask)
		center_pixels = np.array(centroids, dtype=int)


		end_index = current_segment+centroids.shape[1]

		# Resize the arrays if they ended up being too small.
		# Will probably only be called on the last image if at all.
		if (output_images):
			# Pull out sections around the centroid of the superpixel
			image_segments[current_segment:end_index, ...] = \
					gather_regions(image, centroids,
							x_window_size=x_window_size,
							y_window_size=y_window_size)

	    # Pull out the appropriate depth images.
 		for depth_idx in range(0, centroids.shape[1]):
 			segment_depths[current_segment + depth_idx] = \
 					depths[image_idx,
 					       center_pixels[0, depth_idx],
 						   center_pixels[1, depth_idx]]

		# Convert depths to quantized logspace:
		#print 'quantizing depths'
		segment_depths[current_segment:current_segment+centroids.shape[1]] = \
		log_pixelate_values(segment_depths[current_segment:current_segment+centroids.shape[1]],
			bins=depth_bins, min_val=depth_min, max_val=depth_max)

 		#print image_segments[current_segment:end_index, ...].shape
 		#print end_index-current_segment
		for i in range(current_segment,end_index):
			#name = image_output_filepath + '/' + str(image_idx) + '_' + str(i-current_segment) + '.jpg'
			name = str(image_idx) + '_' + str(i-current_segment) + '.jpg'

			# write image
			#print image_segments[i, ...].shape
			#plt.imshow(image_segments[i, ...])
			if output_images:
				scipy.misc.imsave(image_output_filepath + '/' + name,np.transpose(image_segments[i, ...],(0,2,1)))

			out_log.write(name + ' ' + str(int(segment_depths[i])) + '\n')


 		current_segment = current_segment + centroids.shape[1]


def find_neighbors(mask):
    """
    Generates a list of edges that connect neighboring superpixels that
    connect each other from a segmentation mask.
    Returns a Nx2 np.array of edges.
    """
    # Generate a symmetric matrix with a one representing a neighboring
    # connection
    no_pixels=len(np.unique(mask))
    adjacent = np.zeros((no_pixels,no_pixels), dtype=bool)
    for idx_x in range(0, mask.shape[0]-1):
        for idx_y in range(0, mask.shape[1]-1):
            # Check left to right
            if mask[idx_x][idx_y] != mask[idx_x+1][idx_y]:
                adjacent[mask[idx_x][idx_y], mask[idx_x+1][idx_y]] = 1
                adjacent[mask[idx_x+1][idx_y], mask[idx_x][idx_y]] = 1
            # Check Top-Down
            if mask[idx_x][idx_y] != mask[idx_x][idx_y+1]:
                adjacent[mask[idx_x][idx_y], mask[idx_x][idx_y+1]] = 1
                adjacent[mask[idx_x][idx_y+1], mask[idx_x][idx_y]] = 1

    # From the upper triangle of the matrix list out a unique set of
    # edges for the pixels
    edges = np.zeros((np.sum(adjacent) / 2, 2))
    idx_edge = 0
    for idx_x in range(0, adjacent.shape[0]):
        for idx_y in range(idx_x, adjacent.shape[1]):
            if adjacent[idx_x, idx_y]:
                edges[idx_edge, 0] = idx_x
                edges[idx_edge, 1] = idx_y
                idx_edge += 1
    return edges

def preprocess_image(
	image, true_depth=None,
	no_superpixels=200,
	x_window_size=10,
	y_window_size=10,
	depth_bins=None, depth_min = None, depth_max=None):
	"""
	Returns image segments, etc.
	"""
	#print 'loop depth_bins = ', depth_bins
	# Select which images to work with
	# if images == None:
	# 	images = range(0, image_set.shape[0])
	# if type(images) is not tuple:
	# 	images = range(0, images)
	
	# [image_set, depths] = load_dataset(input_filename)
	no_segments = no_superpixels  #* len(images)

	
	image_segments = np.ndarray([no_segments,3,2*x_window_size+1, 2*y_window_size+1])
	segment_depths = np.ndarray(no_segments)
	


	#plt.imshow(image)
	print image.shape
	masks = segment_image(image, no_segments=no_superpixels)
	centroids = calculate_sp_centroids(masks)
	center_pixels = np.array(centroids, dtype=int)

	# Pull out sections around the centroid of the superpixel
	image_segments = \
			gather_regions(image, centroids,
					x_window_size=x_window_size,
					y_window_size=y_window_size)

	# # If provided depth maps, quantize and return those too
	if true_depth is not None:
	#     Pull out the appropriate depth images.
		for depth_idx in range(0, centroids.shape[1]):
			segment_depths[depth_idx] = \
					true_depth[center_pixels[0, depth_idx],
						   	   center_pixels[1, depth_idx]]

	 	# Convert depths to quantized logspace:
	 	#print 'quantizing depths'
	 	segment_depths = log_pixelate_values(segment_depths,
	 		bins=depth_bins, min_val=depth_min, max_val=depth_max)


	if true_depth is not None:
 		return image_segments, masks, segment_depths
 	else:
 		return image_segments, masks



