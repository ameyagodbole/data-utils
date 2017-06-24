import numpy as np
import tensorflow as tf
try:
	from PIL import Image
except ImportError:
	Image = None
import os
from random import shuffle

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_images(filepath, data, labels, as_name=False, reshape=True, im_shape=None,
	dshuffle=True, max_per_file=128, overwrite=True, seed=None):
	"""Dumps images to a TFRecord file.

	The weight file has:
		- `layer_names` (attribute), a list of strings
			(ordered names of model layers).
		- For every layer, a `group` named `layer.name`
			- For every such layer group, a group attribute `weight_names`,
				a list of strings
				(ordered names of weights tensor of the layer).
			- For every weight in the layer, a dataset
				storing the weight value, named after the weight tensor.

	# Arguments
		filepath: String, path to the file to save the data to
		data: The data to dump to HDF5
			  - Can be numpy array of images of shape [samples, height, witdh, channels] type uint8
			  - List of image paths to dump (set as_name to True)
		labels: Array on class labels of shape [samples]
		as_name: Whether data is list of image paths
		reshape: Whether to reshape images
		im_shape: Tuple of image shapes (if reshape)
		dshuffle: Whether to shuffle data
		max_per_file: Number of samples to store per file. If more samples are provided,
						  data is broken into multiple files		
		overwrite: Whether existing HDF5 file is to be overwritten
		seed: Seed value for numpy RandomState

	# Raises
		IOError: If overwrite is False but file exists
		ValueError: Unless otherwise stated, due to invalid argument combinations
	"""
	if( os.path.isfile(filepath) and not overwrite):
		raise IOError('tfr_utils/write_image: File already exists {} and overwrite set to True'.format(filepath))

	if seed is not None:
		np.random.seed(seed)

	if as_name:
		if Image==None:
			raise ImportError('tfr_utils/write_image: PIL.Image not found')
		num_samples = len(data)
	else:
		num_samples = data.shape[0]
	num_files = np.ceil(num_samples/max_per_file)
	if( num_samples>max_per_file ):
		print 'num_samples ({}) > max_per_file ({})\n.Splitting into {} files'.format(num_samples, max_per_file, num_files)
	
	if as_name and reshape:
		if im_shape==None:
			raise ValueError('tfr_utils/write_image: im_shape not assigned in as_name mode')
		else if len(a)<3:
			raise ValueError('tfr_utils/write_image: Invalid value {} passed to im_shape. Needs tuple/list of len 3'.format(im_shape))
	else:
		im_shape = None

	indices = [it for it in range(num_samples)]
	if dshuffle:
		shuffle(indices)

	filename, fileext = os.path.splitext(filepath)
	for it in range(num_files):
		if num_files>1:
			filename = '{}_{:02d}.{}'.format(filename,str(it),fileext)
			writer = tf.python_io.TFRecordWriter(filename)
		else:
			writer = tf.python_io.TFRecordWriter(filepath)

		dlen = min(max_per_file, num_samples-it*max_per_file)
		
		for x in range((it*max_per_file):(it*max_per_file+dlen)):
			if as_name:
				img = Image.open(data[indices[x]])
				if reshape:
					img = img.resize(im_shape[:-1], Image.BILINEAR)
				img = np.array(img)
			else:
				img = data[indices[x]]
			height = img.shape[0]
			width = img.shape[1]
			try:
				channel = img.shape[2]
			except IndexError:
				channel = 1
			annotation = np.int64(labels[indices[x]])
			img_raw = img.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'height': _int64_feature(height),
				'width': _int64_feature(width),
				'channel': _int64_feature(channel),
				'image_raw': _bytes_feature(img_raw),
				'label': _int64_feature(annotation)}))

			writer.write(example.SerializeToString())

		writer.close()

def write_siamese_images(filepath, data1, data2, labels, as_name=False, reshape=True, im_shape=None,
	dshuffle=True, max_per_file=128, overwrite=True, seed=None):
	"""Dumps image pairs to a HDF5 file.

	The weight file has:
		- `layer_names` (attribute), a list of strings
			(ordered names of model layers).
		- For every layer, a `group` named `layer.name`
			- For every such layer group, a group attribute `weight_names`,
				a list of strings
				(ordered names of weights tensor of the layer).
			- For every weight in the layer, a dataset
				storing the weight value, named after the weight tensor.

	# Arguments
		filepath: String, path to the file to save the data to
		data1, data2: The data to dump to HDF5
			  - Can be numpy array of images of shape [samples, height, witdh, channels] of type uint8
			  - List of image paths to dump (set as_name to True)
		labels: Array on class labels of shape [samples]
		as_name: Whether data is list of image paths
		reshape: Whether to reshape images
		im_shape: Tuple of image shapes (if reshape)
		dshuffle: Whether to shuffle data
		max_per_file: Number of samples to store per file. If more samples are provided,
						  data is broken into multiple files		
		overwrite: Whether existing HDF5 file is to be overwritten
		seed: Seed value for numpy RandomState

	# Raises
		IOError: If overwrite is False but file exists
		ValueError: Unless otherwise stated, due to invalid argument combinations
	"""
	if( os.path.isfile(filepath) and not overwrite):
		raise IOError('tfr_utils/write_image: File already exists {} and overwrite set to True'.format(filepath))

	if seed is not None:
		np.random.seed(seed)

	if as_name:
		if Image==None:
			raise ImportError('tfr_utils/write_image: PIL.Image not found')
		num_samples = len(data)
	else:
		num_samples = data.shape[0]
	num_files = np.ceil(num_samples/max_per_file)
	if( num_samples>max_per_file ):
		print 'num_samples ({}) > max_per_file ({})\n.Splitting into {} files'.format(num_samples, max_per_file, num_files)
	
	if as_name and reshape:
		if im_shape==None:
			raise ValueError('tfr_utils/write_image: im_shape not assigned in as_name mode')
		else if len(a)<3:
			raise ValueError('tfr_utils/write_image: Invalid value {} passed to im_shape. Needs tuple/list of len 3'.format(im_shape))
	else:
		im_shape = None

	indices = [it for it in range(num_samples)]
	if dshuffle:
		shuffle(indices)

	filename, fileext = os.path.splitext(filepath)
	for it in range(num_files):
		if num_files>1:
			filename = '{}_{:02d}.{}'.format(filename,str(it),fileext)
			writer = tf.python_io.TFRecordWriter(filename)
		else:
			writer = tf.python_io.TFRecordWriter(filepath)

		dlen = min(max_per_file, num_samples-it*max_per_file)
		
		for x in range((it*max_per_file):(it*max_per_file+dlen)):
			if as_name:
				img1 = Image.open(data1[indices[x]])
				if reshape:
					img1 = img1.resize(im_shape[:-1], Image.BILINEAR)
				img1 = np.array(img1)
				img2 = Image.open(data2[indices[x]])
				if reshape:
					img2 = img2.resize(im_shape[:-1], Image.BILINEAR)
				img2 = np.array(img2)
			else:
				img1 = data1[indices[x]]
				img2 = data2[indices[x]]
			height1 = img1.shape[0]
			width1 = img1.shape[1]
			height2 = img2.shape[0]
			width2 = img2.shape[1]
			try:
				channel1 = img1.shape[2]
			except IndexError:
				channel1 = 1
			try:
				channel2 = img2.shape[2]
			except IndexError:
				channel2 = 1
			annotation = np.int64(labels[indices[x]])
			img1_raw = img1.tostring()
			img2_raw = img2.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'height1': _int64_feature(height1),
				'width1': _int64_feature(width1),
				'channel1': _int64_feature(channel1),
				'image_raw1': _bytes_feature(img1_raw),
				'height2': _int64_feature(height2),
				'width2': _int64_feature(width2),
				'channel2': _int64_feature(channel2),
				'image_raw2': _bytes_feature(img2_raw),
				'label': _int64_feature(annotation)}))

			writer.write(example.SerializeToString())

		writer.close()

def input_read(filenames, batch_size, num_epochs, fshuffle=True, fqsize=32, dshuffle=True, dthreads=2,
	preproc=None, preproc_args=None, siamese=False, im_shape=(224,224,3), seed=None):
	"""Reads input data num_epochs times.
	Args:
		filenames: Files to read from
		batch_size: Number of examples per returned batch.
		num_epochs: Number of times to read the input data, or 0/None to
			 train forever.
		fshuffle: Whether to shuffle file names queue
		fqsize: Size of file names queue 
				(for prediction, it may be best to set fshuffle=False,fqsize=1)
		dshuffle: Whether to shuffle tensors
		dthreads: Worker threads for batching process
		siamese: Whether to generate image pairs
		preproc: Function for preprocessing single example
		preproc_args: Arguments for preprocessing function
		im_shape: Further layers require image shape to be well defined at this stage
				  (needed for Tensor.set_shape)
		seed: Seed for all random functions
	Returns:
		A tuple (question1, question2, is_duplicate)
	"""
	if not num_epochs: num_epochs = None
	
	with tf.name_scope('input'):
		filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs,
			shuffle=fshuffle, seed=seed, capacity=fqsize)

		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)

		if siamese:
			'''
			'height1': _int64_feature(height1),
			'width1': _int64_feature(width1),
			'channel1': _int64_feature(channel1),
			'image_raw1': _bytes_feature(img1_raw),
			'height2': _int64_feature(height2),
			'width2': _int64_feature(width2),
			'channel2': _int64_feature(channel2),
			'image_raw2': _bytes_feature(img2_raw),
			'label': _int64_feature(annotation)
			'''
			features = tf.parse_single_example(
				  serialized_example,
				  features={
					  'height1': tf.FixedLenFeature([], tf.int64),
					  'width1': tf.FixedLenFeature([], tf.int64),
					  'channel1': tf.FixedLenFeature([], tf.int64),
					  'image_raw1': tf.FixedLenFeature([], tf.string),
					  'height2': tf.FixedLenFeature([], tf.int64),
					  'width2': tf.FixedLenFeature([], tf.int64),
					  'channel2': tf.FixedLenFeature([], tf.int64),
					  'image_raw2': tf.FixedLenFeature([], tf.string),
					  'label': tf.FixedLenFeature([], tf.int64),
				  })
			image1 = tf.decode_raw(features['image_raw1'], tf.uint8)
			image2 = tf.decode_raw(features['image_raw2'], tf.uint8)

			image1 = tf.reshape(image1, im_shape)
			image2 = tf.reshape(image2, im_shape)

			if preproc:
				# Call preprocessing function (usually for casting and randomization)
				image1r = preproc(image1, **preproc_args)
				image2r = preproc(image2, **preproc_args)
			else:
				image1r = image1
				image2r = image2
			image1r.set_shape(im_shape)
			image2r.set_shape(im_shape)
			label = tf.cast(features['label'], tf.float32)

			if dshuffle:
				images1, images2, labels = tf.train.shuffle_batch(
					[image1r, image2r, label], batch_size=batch_size,
					capacity=10 + 3 * batch_size,
					min_after_dequeue=2*batch_size,
					num_threads=dthreads,
					seed=seed, allow_smaller_final_batch=True)
				return images1, images2, label
			else:
				images1, images2, labels = tf.train.batch(
					[image1r, image2r, label], batch_size=batch_size,
					num_threads=1,
					capacity=10 + 3 * batch_size,
					allow_smaller_final_batch=True)
				return images1, images2, label
		else:
			'''
			'height': _int64_feature(height),
			'width': _int64_feature(width),
			'channel': _int64_feature(channel),
			'image_raw': _bytes_feature(img_raw),
			'label': _int64_feature(annotation)
			'''
			features = tf.parse_single_example(
				  serialized_example,
				  features={
					  'height': tf.FixedLenFeature([], tf.int64),
					  'width': tf.FixedLenFeature([], tf.int64),
					  'channel': tf.FixedLenFeature([], tf.int64),
					  'image_raw': tf.FixedLenFeature([], tf.string),
					  'label': tf.FixedLenFeature([], tf.int64),
				  })
			image1 = tf.decode_raw(features['image_raw'], tf.uint8)

			image1 = tf.reshape(image1, im_shape)

			if preproc:
				# Call preprocessing function (usually for casting and randomization)
				image1r = preproc(image1, **preproc_args)
			else:
				image1r = image1
			image1r.set_shape(im_shape)
			label = tf.cast(features['label'], tf.float32)

			if dshuffle:
				images1, labels = tf.train.shuffle_batch(
					[image1r, label], batch_size=batch_size,
					capacity=10 + 3 * batch_size,
					min_after_dequeue=2*batch_size,
					num_threads=dthreads,
					seed=seed, allow_smaller_final_batch=True)
				return images1, label
			else:
				images1, labels = tf.train.batch(
					[image1r, label], batch_size=batch_size,
					num_threads=1,
					capacity=10 + 3 * batch_size,
					allow_smaller_final_batch=True)
				return images1, label
