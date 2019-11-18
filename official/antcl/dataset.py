#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""tf.data.Dataset interface to the MNIST dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import shutil
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from sklearn import decomposition
from sklearn import preprocessing

def read32(bytestream):
  """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
  """Validate that filename corresponds to images for the MNIST dataset."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_images, unused
    rows = read32(f)
    cols = read32(f)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))
    if rows != 64 or cols != 76:
      raise ValueError(
          'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
          (f.name, rows, cols))


def check_labels_file_header(filename):
  """Validate that filename corresponds to labels for the MNIST dataset."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    magic = read32(f)
    read32(f)  # num_items, unused
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST file %s' % (magic,
                                                                     f.name))


def download(directory, filename):
  """Download (and unzip) a file from the MNIST dataset if not already done."""
  filepath = os.path.join(directory, filename)
  if tf.io.gfile.exists(filepath):
    return filepath
  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + filename + '.gz'
  _, zipped_filepath = tempfile.mkstemp(suffix='.gz')
  print('Downloading %s to %s' % (url, zipped_filepath))
  urllib.request.urlretrieve(url, zipped_filepath)
  with gzip.open(zipped_filepath, 'rb') as f_in, \
      tf.io.gfile.GFile(filepath, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
  os.remove(zipped_filepath)
  return filepath


def dataset(directory, images_file, labels_file):
  """Download and parse MNIST dataset."""
  '''
  images_file = download(directory, images_file)
  labels_file = download(directory, labels_file)

  check_image_file_header(images_file)
  check_labels_file_header(labels_file)

  def decode_image(image):
    # Normalize from [0, 255] to [0.0, 1.0]
    image = tf.io.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [784])
    return image / 255.0

  def decode_label(label):
    label = tf.io.decode_raw(label, tf.uint8)  # tf.string -> [tf.uint8]
    label = tf.reshape(label, [])  # label is a scalar
    return tf.cast(label, tf.int32)

  images = tf.data.FixedLengthRecordDataset(
      images_file, 28 * 28, header_bytes=16).map(decode_image)
  labels = tf.data.FixedLengthRecordDataset(
      labels_file, 1, header_bytes=8).map(decode_label)
  '''
  # obtain data and labels
  images = np.abs(np.load(directory+images_file))
  labels = np.load(directory+labels_file)
 
  # get data indices of interest for this specific experiment
  idx_phase = np.nonzero(np.in1d(labels[:,1],[1,2]))[0]
  images = images[idx_phase]
  labels = labels[idx_phase]
  idx_trial_type = np.nonzero(np.in1d(labels[:,2],[0,1,2]))[0]
  labels[idx_trial_type,2] = 0
  idx_trial_type = np.nonzero(np.in1d(labels[:,2],[3,4,5]))[0]
  labels[idx_trial_type,2] = 1
  idx_trial_type = np.nonzero(np.in1d(labels[:,2],[6,7,8]))[0]
  labels[idx_trial_type,2] = 2

  # finalize data pre-processing
  if images_file == 'ts_data/x_ts.npy':
      images_tr = np.abs(np.load(directory+'tr_data/x_tr.npy',))
      labels_tr = np.load(directory+'tr_data/y_tr.npy',)
      idx_phase = np.nonzero(np.in1d(labels_tr[:,1],[1,2]))[0]
      images_tr = images_tr[idx_phase]
  else:
      images_tr = images

  images = 1e9*images
  images_tr = 1e9*images_tr
  scaler = preprocessing.StandardScaler().fit(np.reshape(images_tr,(images_tr.shape[0],images_tr.shape[1]*images_tr.shape[2])))
  images = scaler.transform(np.reshape(images,(images.shape[0],images.shape[1]*images.shape[2])))
  images = np.reshape(images,(images.shape[0],images_tr.shape[1],images_tr.shape[2]))
  images = images.astype('float32')
  #images_mean = np.mean(images_tr,axis=0)
  #images -= images_mean
  #images_tr -= images_mean
  #images_std = np.std(images_tr,axis=0)
  #images = images/images_std
  #images_tr = images_tr/images_std
  images = np.expand_dims(images,axis=2)
  images = images[:,:,:,10:35]
  print(np.mean(images,axis=0))
  print(np.std(images,axis=0))
  print(np.mean(images))
  input()
  labels = labels[:,2].astype(np.int)

  return tf.data.Dataset.from_tensor_slices((images, labels))

def dataset_pca(directory, images_file, labels_file):
  # obtain data and labels

  images_tr = np.load(directory+'tr_data/x_tr.npy',)
  labels_tr = np.load(directory+'tr_data/y_tr.npy',)

  idx_phase = np.nonzero(np.in1d(labels_tr[:,1],[1,2]))[0]
  images_tr = images_tr[idx_phase]

  images = np.load(directory+images_file)
  labels = np.load(directory+labels_file)
 
  # get data indices of interest for this specific experiment
  idx_phase = np.nonzero(np.in1d(labels[:,1],[1,2]))[0]
  images = images[idx_phase]
  labels = labels[idx_phase]
  idx_trial_type = np.nonzero(np.in1d(labels[:,2],[0,1,2]))[0]
  labels[idx_trial_type,2] = 0
  idx_trial_type = np.nonzero(np.in1d(labels[:,2],[3,4,5]))[0]
  labels[idx_trial_type,2] = 1
  idx_trial_type = np.nonzero(np.in1d(labels[:,2],[6,7,8]))[0]
  labels[idx_trial_type,2] = 2

  # finalize data pre-processing
  images = images.astype('float32')
  images_tr = images_tr.astype('float32')
  GA = np.mean(images_tr,axis=0)
  pca = decomposition.PCA()
  GA_PCA = pca.fit_transform(GA)
  PC1 = GA_PCA[:,0].reshape(1,64)
  PC2 = GA_PCA[:,1].reshape(1,64)
  PC3 = GA_PCA[:,2].reshape(1,64)
  images = np.transpose(images,axes=(0,2,1))
  im_PC1 = np.dot(images,PC1.T)
  im_PC2 = np.dot(images,PC2.T)
  im_PC3 = np.dot(images,PC3.T)
  images = np.concatenate((im_PC1,im_PC2,im_PC3),axis=2)
  images = np.transpose(images,axes=(0,2,1))
  images = np.expand_dims(images,axis=2)
  images = images/4.1177626e-12
  labels = labels[:,2].astype(np.int)

  return tf.data.Dataset.from_tensor_slices((images, labels))

def train(directory):
  """tf.data.Dataset object for MNIST training data."""
  return dataset(directory, 'tr_data/x_tr.npy',
                 'tr_data/y_tr.npy')


def test(directory):
  """tf.data.Dataset object for MNIST test data."""
  return dataset(directory, 'ts_data/x_ts.npy', 'ts_data/y_ts.npy')
