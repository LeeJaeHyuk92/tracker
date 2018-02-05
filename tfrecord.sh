#!/usr/bin/env bash

# not string
DATA_PATH=/home/jaehyuk/code/own/tracker/data/vot2015
RECORD_PATH=/home/jaehyuk/code/own/tracker/data/tfrecords
NAME=train_1_adj_sample

rm %RECORD_PATH/$NAME.tfrecords
echo -e "\e[31m${NAME} is deleted\e[0m"
python -m src.tfrecord --data_dir $DATA_PATH --out $RECORD_PATH --name $NAME
#python -m src.tfrecord --data_dir /home/jaehyuk/code/own/tracker/data/vot2015 --out /home/jaehyuk/code/own/tracker/data/tfrecords --name train_1_adj_sample

