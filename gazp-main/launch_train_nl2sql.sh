#!/bin/bash

docker run --rm \
  -m4g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python train.py --dataset sparc --name r1 --keep_values --model nl2sql --interactive_eval
