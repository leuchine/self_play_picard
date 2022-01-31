#!/bin/bash

nohup make train_sparc &
nohup make train_sql2text_sparc &

wait

nohup  make self_play_sparc &

wait

nohup  make train_sparc_self_play &