#!/bin/bash

docker run --rm \
  -m4g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python preprocess_nl2sql_sparc.py
