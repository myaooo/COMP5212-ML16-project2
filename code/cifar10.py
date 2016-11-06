# cifar10.py
# project2 - ml16
# Ming Yao
# amended from mnist.py

import os
import sys
import pickle
import time

import numpy as np
from six.moves import urllib
from convnet import *
from utils import *
import tensorflow as tf
import tarfile
import math

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
WORK_DIRECTORY = 'cifar-10'
DATA_DIRECTORY = 'cifar-10/cifar-10-batches-py/'
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_LABELS = 10
TRAIN_SIZE = 50000
TEST_SIZE = 10000
BATCH_SIZE = 100
NUM_EPOCHS = 50
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = int(TRAIN_SIZE/BATCH_SIZE)  # Number of steps between evaluations.


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'cifar-10',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(WORK_DIRECTORY):
        os.makedirs(WORK_DIRECTORY)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    if not os.path.exists(DATA_DIRECTORY):
        tarfile.open(filepath, 'r:gz').extractall(WORK_DIRECTORY)

def extract_data_and_label(eval_data = False):
    """Extract the images into a 4D tensor [image index, y, x, channels].

    Values are normalized
    """
    data_list = []
    label_list = []
    num_images = TRAIN_SIZE
    filenames = [os.path.join(DATA_DIRECTORY, 'data_batch_%d' % i) for i in range(1,6)]
    if eval_data:
        num_images = TEST_SIZE
        filenames = [os.path.join(DATA_DIRECTORY, 'test_batch')]
    for filename in filenames:
        print('Extracting', filename)
        # parse file using pickle
        f = open(filename,'rb')
        dict = pickle.load(f,encoding='bytes')
        # parsing data
        data = dict[b'data']
        data_list.append(data)
        label_list.append(dict[b'labels'])

    data = np.vstack(data_list)
    labels = np.hstack(label_list)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = normalize_data(data)
    data = data.reshape(num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    # reorder dimensions from [num_channel,x,y] into [x,y,num_channel]
    data = data.transpose((0, 2, 3, 1))
    
    return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      np.sum(np.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def CompleteZCAwhitening(train_data, test_data):
    train_data = train_data.reshape(TRAIN_SIZE, IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
    test_data = test_data.reshape(TEST_SIZE, IMAGE_SIZE* IMAGE_SIZE* NUM_CHANNELS)
    data = ZCAwhitening(np.vstack([train_data,test_data]))
    train_data = data[:TRAIN_SIZE,:]
    test_data = data[TRAIN_SIZE:,:]
    train_data = train_data.reshape(TRAIN_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    test_data = test_data.reshape(TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    
    return train_data, test_data

def main(argv=None):  # pylint: disable=unused-argument

    maybe_download_and_extract()
    # Extract it into numpy arrays.
     # Extract it into numpy arrays.
    train_data, train_labels = extract_data_and_label(eval_data=False)
    test_data, test_labels = extract_data_and_label(eval_data=True)
    # train_data, test_data = CompleteZCAwhitening(train_data,test_data)

    num_epochs = NUM_EPOCHS

    # LeNet-5 like Model
    model = ConvNet()
    model.input_data((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), num_label=NUM_LABELS, eval_batch=EVAL_BATCH_SIZE)
    model.add_conv_layer(filter=[5, 5], depth=32, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('max', kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    model.add_conv_layer(filter=[5, 5], depth=64, strides=[1, 1, 1, 1], activation='relu')
    model.add_pool('max', kernel_size=[1, 2, 2, 1], strides=[1, 2, 2, 1])
    model.add_fully_connected(n_units=512, activation='relu')
    model.add_dropout(0.5)
    model.add_fully_connected(n_units=NUM_LABELS,activation='relu')
    model.set_loss(tf.nn.sparse_softmax_cross_entropy_with_logits, reg=5e-4)
    model.set_optimizer('Adam')
    model.init()
    model.train_with_eval(train_data, train_labels, test_data, test_labels, num_epochs, EVAL_FREQUENCY,0.001)


if __name__ == '__main__':
    tf.app.run()
