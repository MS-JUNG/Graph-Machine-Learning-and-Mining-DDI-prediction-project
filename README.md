KAIST, 2024-spring, Graph Machine Learning and Mining

We perform drug-drug interaction prediction in two seperated phases. 


<img src = 'two version detail.PNG'>


Phase 1 : From the data, get the node embeddings.
 run_1_encoder.sh file is the shell script file for phase 1.
 To perform it, type this command.

> bash run_1_encoder.sh

Phase 2 : From the node embeddings made in phase 1, perform a classification.
 run_2_classifier.sh file is the shell script file for phase 2. 
 It consists of sequential command lines for performing classification by 4 types of classifier. (MLP, K-way, attention-MLP, attention-K-way)
 To perform it, type this command. 

> bash run_2_classifier.sh

Result files will be saved in ./model/result directory. 


custom_convs.py is A modified version of the GCNConv module embedded within the PyTorch Geometric.This version introduces an edge feature vector, allowing edge attributes to be learned when aggregating neighbor information.

train_graph_v1.py is for single graph(heterogeneous edge) node embedding learning
train_graph_v2.py is for multiple graphs(homogeneous edge) node embedding learning


The files below use different methods to perform edge prediction (classifier)
through node embeddings trained using two embedding methods (v1, v2).

train_fc_v1.py
train_fc_v2.py
train_kway_v1.py
train_kway_v2.py





