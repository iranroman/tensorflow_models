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


def dataset(directory, images_file):
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
  x_s = np.load(directory+images_file+'_s.npy').astype('float32')
  x_m = np.load(directory+images_file+'_m.npy').astype('float32')
  x_l = np.load(directory+images_file+'_l.npy').astype('float32')

  epoch_size = 75

  images = np.zeros([0,64,epoch_size]).astype('float32')
  labels = np.zeros([0,1])

  for i in np.random.choice(range(epoch_size),2):
      print(i)
      images = np.concatenate((images,np.reshape(x_s[:,:,10+i:-75*4+i].transpose(0,2,1),(x_s.shape[0]*2,epoch_size,64)).transpose(0,2,1)),axis=0)
      labels = np.concatenate((labels,i*np.ones((x_s.shape[0]*2,1))),axis=0)
  for i in np.random.choice(range(epoch_size),2):
      images = np.concatenate((images,np.reshape(x_m[:,:,10+i:-75*7+i].transpose(0,2,1),(x_m.shape[0]*2,epoch_size,64)).transpose(0,2,1)),axis=0)
      labels = np.concatenate((labels,i*np.ones((x_m.shape[0]*2,1))),axis=0)
  for i in np.random.choice(range(epoch_size),2):
      images = np.concatenate((images,np.reshape(x_l[:,:,10+i:-75*10+i].transpose(0,2,1),(x_l.shape[0]*2,epoch_size,64)).transpose(0,2,1)),axis=0)
      labels = np.concatenate((labels,i*np.ones((x_l.shape[0]*2,1))),axis=0)
  # finalize data pre-processing
  images = images.reshape((images.shape[0],images.shape[1]*images.shape[2]))
  images = images/2.14457664857678e-08
  labels = labels.astype(np.int)

  return tf.data.Dataset.from_tensor_slices((images, labels))


def train(directory):
  """tf.data.Dataset object for MNIST training data."""
  return dataset(directory, 'tr_data/x_tr')


def test(directory):
  """tf.data.Dataset object for MNIST test data."""
  return dataset(directory, 'ts_data/x_ts')