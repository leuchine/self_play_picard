#!/bin/bash

docker run --rm \
  -m4g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python train.py --dataset sparc --name r1 --model sql2nl_sparc --fparser exp/nl2sql/r1/best.tar --interactive_eval
