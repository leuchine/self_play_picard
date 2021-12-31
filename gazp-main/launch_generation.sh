#!/bin/bash

docker run --rm \
  -m8g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python generate_sql.py --use_sql_prob_error_analysis --num 50000 --fout gen/gen1.json --resume exp/sql2nl_sparc/r1/best.tar --fparser exp/nl2sql/r1/best.tar --dataset sparc --tables data/sparc/tables.json --db data/sparc/database --model sql2nl_sparc --ftrain data/sparc/train.json
