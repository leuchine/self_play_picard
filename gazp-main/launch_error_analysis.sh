#!/bin/bash

docker run --rm \
  -m8g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python error_analysis.py --dataset sparc --tables data/sparc/tables.json --db data/sparc/database --model sql2nl_sparc --ftrain data/sparc/train.json
