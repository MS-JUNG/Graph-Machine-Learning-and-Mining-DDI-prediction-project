import pandas as pd 
import scanpy as sc 
import torch 
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch.nn.functional as F
# Import custom GCNConv layer
from custom_convs import GCNConv
from torch_geometric.utils import *
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict
import random

seed_num = 42
print("seed number : {}".format(seed_num))

torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)

for num in tqdm(range(1,5)):
    # load similarity matrices of 4 features (572x572)
    dataset = pd.read_csv(f'../data/featuers_m{num}.txt', header = None, sep = ' ')
    
    dataset_m1 = dataset.iloc[:,1:]
    dataset_m1 = torch.tensor(np.array(dataset_m1))

    # load label(edge type) - node - node data
    label = pd.read_csv('../data/full_pos2.txt', header=None, sep = ' ')
    edge =  np.array(label).tolist()
    edge_dict = defaultdict(list)

    # load label(edge type) - node - node - weight data
    full_pos2 = pd.read_csv('../data/full_pos2_weight.csv', sep = ',', skiprows=0)
    full_pos2_list = np.array(full_pos2).tolist()
    edge_weight = [] # list of edge weight

    for f in full_pos2_list:
        edge_weight.append(f[3])
    print(edge_weight[0])

    edge_P = [] # list of edge pairs
    edge_L = [] # list of edge labels
    
    for edg in edge[1:]:
        edge_dict[int(edg[0])].append([int(edg[1]),int(edg[2])])
        edge_P.append([int(edg[1]),int(edg[2])])
        edge_L.append(int(edg[0]))
   

    edges = np.array(label).tolist()

    # dimension of node embedding
    dim = 256 

    ### Define GCN model for node embedding
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(572, 400, edge_dim =66)   # Specify edge_dim (one-hot)
            self.conv2 = GCNConv(400, dim, edge_dim =66)    
            self.linear = nn.Linear(dim *2 ,66)   # 66 classes for multi-class classification (0-64 + no-edge class 65)


        def forward(self, data,node_pair):
            # x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr 
            x, edge_index, edge_attr, edge_weight = data.x, data.edge_index, data.edge_attr, data.edge_weight # If you want to remove the edge weight element, replace it with the upper line
            x = x.float()

            # x = self.conv1(x, edge_index, edge_attr = edge_attr) 
            x = self.conv1(x, edge_index, edge_attr = edge_attr, edge_weight=edge_weight) # If you want to remove the edge weight element, replace it with the upper line
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            # x = self.conv2(x,edge_index, edge_attr = edge_attr)
            x = self.conv2(x,edge_index, edge_attr = edge_attr, edge_weight=edge_weight) # If you want to remove the edge weight element, replace it with the upper line
            drug_emb = x 

            x = torch.cat((x[node_pair[0]],x[node_pair[1]]), dim =1) # Concatenate the node embeddings for each pair
            x = self.linear(x)

            return x, drug_emb
        
    
    ### Function to generate embedding data
    def embedding_data_generate():
        
        edge = edge_P 
        label = edge_L
        
        edge = torch.tensor(edge) 
        label = torch.tensor(label)

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(edge, label, edge_weight, test_size= 0.15,random_state=42) # train:val:test = 0.7:0.15:0.15
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_train, y_train, w_train, test_size= 0.15,random_state=42) 

        # Define edge attributes as one-hot vectors -> more advantageous / edge attributes = one-hot vectors of edge types
        edge_attr_train = F.one_hot(y_train, num_classes=66).float() # Ensure edge_attr has shape [num_edges, edge attributes]
        edge_attr_val = F.one_hot(y_val, num_classes=66).float()
        edge_attr_test = F.one_hot(y_test, num_classes=66).float()    

        # Define edge weights which is from Pagerank
        edge_weight_train = torch.tensor(w_train, dtype=torch.float)
        edge_weight_val = torch.tensor(w_val, dtype=torch.float)
        edge_weight_test = torch.tensor(w_test, dtype=torch.float)
        
        # Generate negative samples for training
        neg_train = negative_sampling(X_train.t().contiguous(),572,len(X_train)) 
        t_node_pair = torch.cat([X_train.t().contiguous(),neg_train],dim =1) 
        t_label_n = torch.full((neg_train.size(1),), 65, dtype=torch.long) # for negative sampling, labeling with 65
        y_train = torch.cat([y_train.long(),t_label_n],dim=0) 
    
        # Generate negative samples for validation
        neg_val = negative_sampling(X_val.t().contiguous(),572,len(X_val))
        v_node_pair = torch.cat([X_val.t().contiguous(),neg_val],dim =1)
        v_label_n = torch.full((neg_val.size(1),), 65, dtype=torch.long)
        y_val = torch.cat([y_val.long(),v_label_n],dim=0)
        
        # Generate negative samples for testing
        neg_test = negative_sampling(X_test.t().contiguous(),572,len(X_test))
        Test_node_pair = torch.cat([X_test.t().contiguous(),neg_test],dim =1)
        Test_node_pair_out = X_test.t().contiguous()
        t_label_n = torch.full((neg_test.size(1),), 65, dtype=torch.long)
        y_test = torch.cat([y_test.long(),t_label_n],dim=0)    

        # train_graph = Data(x=dataset_m1, edge_index = X_train.t().contiguous(), label = y_train, edge_attr=edge_attr_train)
        # val_graph = Data(x=dataset_m1, edge_index = X_val.t().contiguous(), label = y_val, edge_attr=edge_attr_val)
        # test_graph = Data(x=dataset_m1, edge_index = X_test.t().contiguous(), label = y_test, edge_attr=edge_attr_test)
        
        # If you want to remove the edge weight element, replace it with the upper line
        train_graph = Data(x=dataset_m1, edge_index = X_train.t().contiguous(), label = y_train, edge_attr=edge_attr_train, edge_weight=edge_weight_train) 
        val_graph = Data(x=dataset_m1, edge_index = X_val.t().contiguous(), label = y_val, edge_attr=edge_attr_val, edge_weight=edge_weight_val)
        test_graph = Data(x=dataset_m1, edge_index = X_test.t().contiguous(), label = y_test, edge_attr=edge_attr_test, edge_weight=edge_weight_test)
  
        return train_graph, val_graph, test_graph, t_node_pair,v_node_pair, Test_node_pair, Test_node_pair_out

    ### Function to train and validate the model
    def train_val(train_graph, val_graph,t_node_pair,v_node_pair):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Net().to(device)
        train_graph = train_graph.to(device)
        val_graph = val_graph.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        f1_list = []

        for epoch in range(120):
            model.train()
            
            optimizer.zero_grad()
            out,_ = model(train_graph,t_node_pair)
            out= out.squeeze(1)

            loss = F.cross_entropy(out, train_graph.label)
            loss.backward()
            optimizer.step()
            

            model.eval()

            with torch.no_grad():
                out,drug_emb = model(train_graph,v_node_pair)
                out = out.squeeze(1)
                loss_val = F.cross_entropy(out, val_graph.label)

                _, predicted = torch.max(out, dim=1)
            
                acc_score_r = accuracy_score(predicted.cpu(),val_graph.label.cpu())
                f1_list.append(acc_score_r)
        
            
        print("Max validation accuracy: ", max(f1_list))
        return model, drug_emb

            

    test = []
    
    train_graph, val_graph, test_graph, t_node_pair, v_node_pair, Test_node_pair, Test_node_pair_out = embedding_data_generate()    

    model,drug_emb = train_val(train_graph, val_graph,t_node_pair,v_node_pair) 
    drug_emb = drug_emb.detach().cpu().numpy()
    df = pd.DataFrame(drug_emb)
    Test_node_pairs = Test_node_pair_out.t().contiguous().tolist()

    for i in range(len(Test_node_pairs)):
        test.append(Test_node_pairs[i])
    df.to_csv(f'../node_emb_v1/emb_m{num}.csv', index=False)
        
    test = np.array(test)
    test = pd.DataFrame(test)

    test.to_csv(f'../data/test.csv', index=False)