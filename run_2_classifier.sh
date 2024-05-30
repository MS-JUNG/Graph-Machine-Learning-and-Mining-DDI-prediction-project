#!/bin/sh

cd ./model
mkdir results
ORDER="GCN GAT GraphSAGE GIN"

###  4 types of classifiers 

# 1. simple MLP
python train_fc_v1.py > results/result_fc_v1.out
python train_fc_v2.py > results/result_fc_v2.out

# 2. simple K-Way edge predictor
python train_kway_v1.py > results/result_kway_v1.out
for i in $ORDER
do
 python train_kway_v2.py $i > results/result_kway_v2_${i}.out
done

# 3. Self-attention attached MLP
python train_att_fc_kway_v1.py MLP > results/result_att_MLP_v1.out
for i in $ORDER
do
   python train_att_fc_kway_v2.py MLP $i > results/result_att_MLP_v2_${i}.out
done

# 4. Self-attention attached K-way 
python train_att_fc_kway_v1.py kway > results/result_att_kway_v1.out
for i in $ORDER
do
   python train_att_fc_kway_v2.py kway $i > results/result_att_kway_v2_${i}.out
done
