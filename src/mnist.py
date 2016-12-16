#
# project: project 2
# file: mnist.py
# author: MING Yao
#

import tensorflow as tf
import numpy as np
from convnet.convnet import ConvNet
import gzip
import os
import sys
from six.moves import urllib

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
# VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 100
NUM_EPOCHS = 20
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 600  # Number of steps between evaluations.

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
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


def main(argv=None):  # pylint: disable=unused-argument

    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    num_epochs = NUM_EPOCHS

    # LeNet-5 like Model
    model = ConvNet('mnist')
    model.push_input_layer(dshape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    model.push_conv_layer(filter_size=[5, 5], out_channels=32, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_conv_layer(filter_size=[5, 5], out_channels=64, strides=[1, 1], activation='relu')
    model.push_pool_layer('max', kernel_size=[1, 2, 2, 1], strides=[2, 2])
    model.push_flatten_layer()
    model.push_fully_connected_layer(out_channels=512, activation='relu')
    model.push_dropout_layer(0.5)
    model.push_fully_connected_layer(out_channels=NUM_LABELS, activation='linear')

    model.set_loss('sparse_softmax')
    model.set_regularizer('l2', 5e-4)
    model.set_learning_rate(0.001)
    model.set_optimizer('Adam')
    model.set_data(train_data[:10000, ...], train_labels[:10000,...], test_data, test_labels)
    model.compile()
    model.train(BATCH_SIZE, 50, EVAL_FREQUENCY)


if __name__ == '__main__':
    tf.app.run()
