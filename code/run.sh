#!/bin/bash

source ~/tensorflow/bin/activate

echo "running mnist.py"
python3 mnist.py
echo "mnist.py done"

deactivate