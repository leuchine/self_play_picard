#!/bin/bash

docker run --rm \
  -m8g \
  --runtime=nvidia \
  -v ${PWD}:/workspace \
  test python generate_sql.py --num 20000 --fout gen/gen1.json --resume exp/sql2nl_cosql/r1/best.tar --fparser exp/nl2sql_cosql/r1/best.tar --dataset cosql --tables data/cosql/tables.json --db data/cosql/database --model sql2nl_cosql --ftrain data/cosql/train.json
