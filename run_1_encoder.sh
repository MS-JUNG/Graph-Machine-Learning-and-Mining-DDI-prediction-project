#!/bin/sh

cd ./model

python pagerank.py
python train_graph_v1.py
python train_graph_v2.py
