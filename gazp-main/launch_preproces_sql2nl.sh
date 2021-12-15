#!/bin/bash

docker run --rm \
  -m4g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python preprocess_sql2nl_sparc.py
