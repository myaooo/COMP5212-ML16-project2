# COMP5212-ML16-project2

Author: MING Yao, No.: 20365468

## Environment Set Up

### Setting Up TensorFlow

Set up TensorFlow environment using virtualenv following the instructions on https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html#virtualenv-installation 

* Python Version: 3.5.2
* TensorFlow Version: r0.11, CPU only

### Others

Install Scipy package to use io libraries for .mat files.

`pip install scipy`

## Demos

Before running demos, run `source ~/tensorflow/bin/activate` to enable tensorflow running environment.

For MNIST, run `python3 mnist.py`.

For CIFAR-10, run `python3 cifar10.py`

For Caltech 101 with LeNet model, run `python3 caltech101lenet.py`.

For Caltech 101 with NIN model, run `python3 caltech101nin.py`.

To disable TensorFlow environment, run `deactivate`.

You can directly run `bash ./run.sh` with bash to run all the demos sequentially.

## Results

Results are in the directory `/code/output`
