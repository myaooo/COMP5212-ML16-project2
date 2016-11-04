# caltech101lenet.py
# project2 - ml16
# Ming Yao

import os
import sys
import time

import numpy as np
from convnet import *
import tensorflow as tf
import scipy.io as sio

DATA_DIRECTORY = 'caltech'
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 102 # 101 categories with a background
TRAIN_SIZE = 8000
TEST_SIZE = 1144
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 100
NUM_EPOCHS = 100
# EVAL_BATCH_SIZE = 104
EVAL_FREQUENCY = 80  # Number of steps between evaluations.


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'caltech',
                           """Path to the Caltech101 data directory.""")

adjust = 1.0/math.sqrt(IMAGE_SIZE*IMAGE_SIZE*NUM_CHANNELS)

def whiten_image(image):
    stddev = np.std(image)
    adjust_stddev = max(stddev, adjust)
    mean = np.mean(image)
    return (image -mean)/adjust_stddev

def extract_data_and_label(eval_data = False):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    data_list = []
    label_list = []
    num_images = TRAIN_SIZE
    filename = os.path.join(DATA_DIRECTORY, 'data_batch_1.mat')
    if eval_data:
        num_images = TEST_SIZE
        filename = os.path.join(DATA_DIRECTORY, 'test_batch.mat')

    print('Extracting', filename)
    dict = sio.loadmat(filename)
    # parsing data
    data = dict['data']
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    labels = dict['labels']
    labels = labels.reshape(num_images)
    for i in range(0,num_images):
        data[i,:,:,:] = whiten_image(data[i,:,:,:])
    return data, labels

def main(argv=None):  # pylint: disable=unused-argument

    # Extract it into numpy arrays.
    train_data, train_labels = extract_data_and_label(eval_data=False)
    test_data, test_labels = extract_data_and_label(eval_data=True)

    num_epochs = NUM_EPOCHS

    # LeNet-5 like Model
    model = ConvNet()
    model.input_data((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), num_label=NUM_LABELS, eval_batch=BATCH_SIZE)
    model.add_conv_layer(filter=[5, 5], depth=32, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('max', kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    model.add_conv_layer(filter=[5, 5], depth=64, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('max', kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    model.add_fully_connected(n_units=512, activation='relu')
    model.add_dropout(0.5)
    model.add_fully_connected(n_units=NUM_LABELS,activation='relu')
    model.set_loss(tf.nn.sparse_softmax_cross_entropy_with_logits, reg=1e-3)
    model.set_optimizer('Adam')
    model.init()
    model.train_with_eval(train_data, train_labels, test_data, test_labels, num_epochs, EVAL_FREQUENCY, 0.001)


if __name__ == '__main__':
    tf.app.run()
