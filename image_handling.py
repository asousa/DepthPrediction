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
	cuts = np.logspace(np.log(min_val), np.log(max_val), num=bins, base=np.e)
	array_vals = np.array(array[:], dtype=int)
	val = np.reshape(np.digitize(array_vals.flatten(), cuts), array.shape)
	return val

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

	current_segment = 0
	for image_idx in images:
		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
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

	return image_segments[0:current_segment, ...], segment_depths[0:current_segment, ...]



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

	output_file = h5py.File(output_filename, 'w')
	image_segments = output_file.create_dataset("data",
		(no_segments, image_set.shape[1], 2 * x_window_size + 1, 2 * y_window_size + 1),
		chunks=(1, image_set.shape[1], 2 * x_window_size + 1, 2 * y_window_size + 1))

	segment_depths = output_file.create_dataset("label", (no_segments, 1), chunks=True)
	segment_image_index = output_file.create_dataset("image", (no_segments, 1), chunks=True)
	segment_superpixel_index = output_file.create_dataset("pixel", (no_segments, 1), chunks=True)

	current_segment = 0
	for image_idx in images:
		image = np.array(image_set[image_idx, ...])
		mask = segment_image(image, no_segments=no_superpixels)
		centroids = calculate_sp_centroids(mask)
		center_pixels = np.array(centroids, dtype=int)

		# Resize the arrays if they ended up being too small.
		# Will probably only be called on the last image if at all.
		end_index = current_segment+centroids.shape[1]
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

