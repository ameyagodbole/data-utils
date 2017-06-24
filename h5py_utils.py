import h5py
import numpy as np
import os
from random import shuffle
import threading
try:
	from PIL import Image
except ImportError:
	Image = None
try:
	import tensorflow as tf
except ImportError:
	tf = None

def write_images(filepath, data, labels, as_name=False,  im_shape=None,
	dshuffle=True, max_per_file=128, overwrite=True, dtype=np.float32, seed=None):
	"""Dumps images to a HDF5 file.

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
			  - Can be numpy array of images of shape [samples, height, witdh, channels]
			  - List of image paths to dump (set as_name to True)
		labels: Array on class labels of shape [samples]
		as_name: Whether data is list of image paths
		im_shape: Tuple of image shapes (if as_name)
		dshuffle: Whether to shuffle data
		max_per_file: Number of samples to store per file. If more samples are provided,
						  data is broken into multiple files		
		overwrite: Whether existing HDF5 file is to be overwritten
		dtype: Data type of stored data
		seed: Seed value for numpy RandomState

	# Raises
		IOError: If overwrite is False but file exists
		ValueError: Unless otherwise stated, due to invalid argument combinations
	"""
	if( os.path.isfile(filepath) and not overwrite):
		raise IOError('h5py_utils/write_image: File already exists {} and overwrite set to True'.format(filepath))

	if seed is not None:
		np.random.seed(seed)

	if as_name:
		if Image==None:
			raise ImportError('h5py_utils/write_image: PIL.Image not found')
		num_samples = len(data)
	else:
		num_samples = data.shape[0]
	num_files = np.ceil(num_samples/max_per_file)
	if( num_samples>max_per_file ):
		print 'num_samples ({}) > max_per_file ({})\n.Splitting into {} files'.format(num_samples, max_per_file, num_files)
	
	if as_name:
		if im_shape==None:
			raise ValueError('h5py_utils/write_image: im_shape not assigned in as_name mode')
		else if len(im_shape)<3:
			raise ValueError('h5py_utils/write_image: Invalid value {} passed to im_shape. Needs tuple/list of len 3'.format(im_shape))
	else:
		im_shape = data.shape[1:]

	indices = [it for it in range(num_samples)]
	if dshuffle:
		shuffle(indices)

	filename, fileext = os.path.splitext(filepath)
	for it in range(num_files):
		if num_files>1:
			filename = '{}_{:02d}.{}'.format(filename,str(it),fileext)
			f = h5py.File(filename, "w")
		else:
			f = h5py.File(filepath, "w")

		dlen = min(max_per_file, num_samples-it*max_per_file)
		dset = f.create_dataset("data", (dlen,)+tuple(im_shape), dtype=dtype, chunks=True, compression="gzip")
		dset.attrs['im_shape'] = im_shape
		dset.attrs['len'] = dlen
		dset.attrs['set_dtype'] = dtype
		dset.attrs['partial'] = 0 if (num_files==1) else 1
		lset = f.create_dataset("labels", (dlen), dtype=dtype, chunks=True, compression="gzip")
		lset.attrs['len'] = dlen
		lset.attrs['set_dtype'] = dtype
		lset.attrs['partial'] = 0 if (num_files==1) else 1

		if as_name:
			imdata = np.zeros(dset.shape)
			for x in range((it*max_per_file):(it*max_per_file+dlen)):
				im = Image.open(data[indices[x]])
				im = im.resize(im_shape[:-1], Image.BILINEAR)
				imdata[x-it*max_per_file,...] = np.asarray(im[...])
			dset[:,...] = imdata[:,...].astype(dtype)
			lset[...] = labels[indices[ (it*max_per_file):(it*max_per_file+dlen)] ].astype(dtype)
		else:
			dset[:,...] = data[indices[ (it*max_per_file):(it*max_per_file+dlen)],... ].astype(dtype)
			lset[...] = labels[indices[ (it*max_per_file):(it*max_per_file+dlen)] ].astype(dtype)

		f.flush()
		f.close()

def write_siamese_images(filepath, data1, data2, labels, as_name=False,  im_shape=None,
	dshuffle=True, max_per_file=128, overwrite=True, dtype=np.float32, seed=None):
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
			  - Can be numpy array of images of shape [samples, height, witdh, channels]
			  - List of image paths to dump (set as_name to True)
		labels: Array on class labels of shape [samples]
		as_name: Whether data is list of image paths
		im_shape: Tuple of image shapes (if as_name)
		dshuffle: Whether to shuffle data
		max_per_file: Number of samples to store per file. If more samples are provided,
						  data is broken into multiple files		
		overwrite: Whether existing HDF5 file is to be overwritten
		dtype: Data type of stored data
		seed: Seed value for numpy RandomState

	# Raises
		IOError: If overwrite is False but file exists
		ValueError: Unless otherwise stated, due to invalid argument combinations
	"""
	if( os.path.isfile(filepath) and not overwrite):
		raise IOError('h5py_utils/write_image: File already exists {} and overwrite set to True'.format(filepath))

	if seed is not None:
		np.random.seed(seed)

	if as_name:
		if Image==None:
			raise ImportError('h5py_utils/write_image: PIL.Image not found')
		num_samples = len(data1)
	else:
		num_samples = data1.shape[0]
	num_files = np.ceil(num_samples/max_per_file)
	if( num_samples>max_per_file ):
		print 'num_samples ({}) > max_per_file ({})\n.Splitting into {} files'.format(num_samples, max_per_file, num_files)
	
	if as_name:
		if im_shape==None:
			raise ValueError('h5py_utils/write_image: im_shape not assigned in as_name mode')
		else if len(a)<3:
			raise ValueError('h5py_utils/write_image: Invalid value {} passed to im_shape. Needs tuple/list of len 3'.format(im_shape))
	else:
		im_shape = data.shape[1:]

	indices = [it for it in range(num_samples)]
	if dshuffle:
		shuffle(indices)

	filename, fileext = os.path.splitext(filepath)
	for it in range(num_files):
		if num_files>1:
			filename = '{}_{:02d}.{}'.format(filename,str(it),fileext)
			f = h5py.File(filename, "w")
		else:
			f = h5py.File(filepath, "w")

		dlen = min(max_per_file, num_samples-it*max_per_file)
		dseta = f.create_dataset("data1", (dlen,)+tuple(im_shape), dtype=dtype, chunks=True, compression="gzip")
		dseta.attrs['im_shape'] = im_shape
		dseta.attrs['len'] = dlen
		dseta.attrs['set_dtype'] = dtype
		dseta.attrs['partial'] = 0 if (num_files==1) else 1
		dsetb = f.create_dataset("data2", (dlen,)+tuple(im_shape), dtype=dtype, chunks=True, compression="gzip")
		dsetb.attrs['im_shape'] = im_shape
		dsetb.attrs['len'] = dlen
		dsetb.attrs['set_dtype'] = dtype
		dsetb.attrs['partial'] = 0 if (num_files==1) else 1
		lset = f.create_dataset("labels", (dlen), dtype=dtype, chunks=True, compression="gzip")
		lset.attrs['len'] = dlen
		lset.attrs['set_dtype'] = dtype
		lset.attrs['partial'] = 0 if (num_files==1) else 1

		if as_name:
			im1data = np.zeros(dseta.shape)
			im2data = np.zeros(dsetb.shape)
			for x in range((it*max_per_file):(it*max_per_file+dlen)):
				im = Image.open(data1[indices[x]])
				im = im.resize(im_shape[:-1], Image.BILINEAR)
				im1data[x-it*max_per_file,...] = np.asarray(im[...])
				im = Image.open(data2[indices[x]])
				im = im.resize(im_shape[:-1], Image.BILINEAR)
				im2data[x-it*max_per_file,...] = np.asarray(im[...])
			dseta[:,...] = im1data[:,...].astype(dtype)
			dsetb[:,...] = im2data[:,...].astype(dtype)
			lset[...] = labels[indices[ (it*max_per_file):(it*max_per_file+dlen)] ].astype(dtype)
		else:
			dseta[:,...] = data1[indices[ (it*max_per_file):(it*max_per_file+dlen)],... ].astype(dtype)
			dsetb[:,...] = data2[indices[ (it*max_per_file):(it*max_per_file+dlen)],... ].astype(dtype)
			lset[...] = labels[indices[ (it*max_per_file):(it*max_per_file+dlen)] ].astype(dtype)

		f.flush()
		f.close()

class HDF5ImageReader():
	"""Reads input data num_epochs times.
	NOTE:
	Not suitable for parallel batch generation
	Use only 1 worker

	Returns:
		batch_sze tuples (image, label) per call to next
	"""

	def __init__(self, files, batch_size, num_epochs,
		fshuffle=True, dshuffle=True, siamese=False, seed=None, verbose=False):
		'''
		Args:
			files: List of input files
			batch_size: Number of examples per returned batch.
			num_epochs: Number of times to read the input data, or 0/None to
				 train forever.
			fshuffle: Whether to shuffle files
			dshuffle: Whether to shuffle data
			siamese: Whether to generate image pairs
			seed: Value for tensorflow filereader
			verbose: Print debugging output 
		'''
		if verbose:
			print 'HDF5ImageReader.__init__'

		if seed is not None:
			np.random.seed(seed)

		self.files = files
		self.fileqsize = len(self.files)
		self.batch_size = batch_size
		self.epochs = num_epochs
		self.fshuffle = fshuffle
		if fshuffle:
			shuffle(self.files)
		self.fcursor = 0
		self.dshuffle = dshuffle
		self.dcursor = 0
		self.siamese = siamese
		self.verbose = verbose
		# self.lock = threading.Lock()
		if self.siamese:
			# TODO: Implement next_siamese
			self.next = self.next_siamese
		else:
			self.next = self.next_batch
		# Set in first batch per file
		self.file = None
		self.size = 0
		self.indices = []
		self.stop = False

		if verbose:
			print 'HDF5ImageReader.__init__ : Proper init'

	def get_next_file(self):
		self.fcursor += 1
		if self.fcursor >= self.fileqsize:
			if self.epochs:
				self.epochs -= 1
			if self.epochs == 0:
				self.stop = True
			self.fcursor = 0
		self.dcursor = 0
		self.file = h5py.File(files[fcursor], "r")
		self.size = self.file['data'].attrs['len']
		self.indices = [x for x in range(self.size)]
		if self.dshuffle:
			shuffle(self.indices)

	def next_batch(self):
		n = self.batch_size
		# First batch
		if self.file == None:
			self.get_next_file()

		# Finish file
		if self.dcursor+n > self.size:
			indices = self.indices[self.dcursor:]
			indices.sort()
			dset = self.file['data'][indices,...]
			lset = self.file['labels'][indices,...]
			self.get_next_file()
			indices = self.indices[self.dcursor:self.dcursor+n-len(dset)]
			indices.sort()
			dset1 = self.file['data'][indices,...]
			lset1 = self.file['labels'][indices,...]
			dset = np.concatenate((dset,dset1), axis=0).astype(np.float32)
			lset = np.concatenate((lset,lset1), axis=0).astype(np.float32)
		else:
			indices = self.indices[self.dcursor:self.dcursor+n]
			indices.sort()
			dset = self.file['data'][indices,...]
			lset = self.file['labels'][indices,...]

		self.dcursor += n
		return (dset,lset)

	def next_siamese:
		n = self.batch_size
		# First batch
		if self.file == None:
			self.get_next_file()

		# Finish file
		if self.dcursor+n > self.size:
			indices = self.indices[self.dcursor:]
			indices.sort()
			dseta = self.file['data1'][indices,...]
			dsetb = self.file['data2'][indices,...]
			lset = self.file['labels'][indices,...]
			self.get_next_file()
			indices = self.indices[self.dcursor:self.dcursor+n-len(dset)]
			indices.sort()
			dset1a = self.file['data1'][indices,...]
			dset1b = self.file['data2'][indices,...]
			lset1 = self.file['labels'][indices,...]
			dseta = np.concatenate((dseta,dset1a), axis=0).astype(np.float32)
			dsetb = np.concatenate((dsetb,dset1b), axis=0).astype(np.float32)
			lset = np.concatenate((lset,lset1), axis=0).astype(np.float32)
		else:
			indices = self.indices[self.dcursor:self.dcursor+n]
			indices.sort()
			dseta = self.file['data1'][indices,...]
			dsetb = self.file['data2'][indices,...]
			lset = self.file['labels'][indices,...]

		self.dcursor += n
		return (dseta, dsetb, lset)