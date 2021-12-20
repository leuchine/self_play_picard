#!/bin/bash

nohup make train_cosql &
nohup make train_sql2text_cosql &

wait

nohup  make self_play_cosql &

wait

nohup  make train_cosql_self_play &