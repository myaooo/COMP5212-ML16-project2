# ML 2016, HKUST
# Project 2
# MING Yao
# yaoming.thu@gmail.com

from __future__ import absolute_import, division, print_function

import gzip
import os
import sys
import time

import numpy

import tensorflow as tf
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import urllib


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


# def fake_data(num_images):
#   """Generate a fake dataset that matches the dimensions of MNIST."""
#   data = numpy.ndarray(
#       shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
#       dtype=numpy.float32)
#   labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
#   for image in xrange(num_images):
#     label = image % 2
#     data[image, :, :, 0] = label - 0.5
#     labels[image] = label
#   return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])



def main(argv=None): # pylint: disable=unused-argument
    # test function
    print('LeNet: Running Test.')


if __name__ == "__main__":
    tf.app.run()
