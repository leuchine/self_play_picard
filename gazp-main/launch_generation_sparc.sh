#!/bin/bash

docker run --rm \
  -m8g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python generate_sql.py --num 150 --fout gen/gen1.json --resume exp/sql2nl_cosql/cosql_sql2nl.tar --fparser exp/nl2sql_cosql/cosql_nl2sql.tar --dataset sparc --tables data/sparc/tables.json --db data/sparc/database --model sql2nl_cosql --ftrain data/sparc/train.json
