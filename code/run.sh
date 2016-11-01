#!/bin/bash

source ~/tensorflow/bin/activate

echo "running mnist_new.py"
python3 mnist_new.py
echo "mnist_new.py done"
echo "running cifar10.py"
python3 cifar10.py
echo "cifar10.py done"
echo "running caltech101lenet.py"
python3 caltech101lenet.py
echo "caltech101lenet.py done"
echo "running caltech101nin.py"
python3 caltech101nin.py
echo "caltech101nin.py done"

deactivate