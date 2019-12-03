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
  # obtain data and labels
  images = np.abs(np.load(directory+images_file))[:,:,10:35]
  images = np.delete(images,[32,42,23,31,41,33,59,63,14,22],axis=1)
  images = 1e6*images
  images = images.astype('float32')
  labels = np.load(directory+labels_file)
 
  # get data indices of interest for this specific experiment
  idx_vis = np.nonzero(np.in1d(labels[:,1],[0]))[0]
  idx_ini = np.nonzero(np.in1d(labels[:,1],[1]))[0]
  idx_ant = np.nonzero(np.in1d(labels[:,1],[2]))[0]
  idx_cha = np.nonzero(np.in1d(labels[:,1],[3]))[0]
  idx_pul = np.concatenate((idx_ini,idx_ant,idx_cha),axis=0)
  n_vis = idx_vis.shape[0]
  n_pul = idx_pul.shape[0]
  ran_i_pul = np.random.choice(n_pul,n_vis,replace=False) 
  idx_data = np.concatenate((idx_vis,idx_pul[ran_i_pul]),axis=0)
  images = images[idx_data]
  labels[idx_vis,1] = 0
  labels[idx_pul[ran_i_pul],1] = 1
  labels = labels[idx_data]
  images = np.expand_dims(images,axis=2)
  #print(images)
  #print(images.shape)
  #input()
  labels = labels[:,1].astype(np.int)

  return tf.data.Dataset.from_tensor_slices((images, labels))

def train(directory):
  """tf.data.Dataset object for MNIST training data."""
  return dataset(directory, 'tr_data/x_tr.npy',
                 'tr_data/y_tr.npy')

def test(directory):
  """tf.data.Dataset object for MNIST test data."""
  return dataset(directory, 'ts_data/x_ts.npy', 'ts_data/y_ts.npy')
