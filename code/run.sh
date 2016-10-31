#!/bin/bash

source ~/tensorflow/bin/activate

echo "running mnist_new.py"
python3 mnist_new.py
echo "mnist_new.py done"

deactivate