#!/bin/sh

cd ./model
mkdir results
ORDER="GCN GAT GraphSAGE GIN"

###  4 types of classifiers 

# 1. simple MLP
python train_fc_v1.py > results/result_fc_v1.out
python train_fc_v2.py > results/result_fc_v2.out

