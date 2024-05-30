import pandas as pd 
import scanpy as sc 
import torch 
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.utils import *
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict
from torch.nn import Linear, ReLU, Sequential
import random
import torch.backends.cudnn as cudnn

seed_num = 42
print("seed number : {}".format(seed_num))

torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)

for i in tqdm(range(1,5)):
    dataset = pd.read_csv(f'../data/featuers_m{i}.txt', header = None, sep = ' ')
    dataset_m1 = dataset.iloc[:,1:]
    weight = pd.read_csv('../data/full_pos2_weight.csv', sep = ',')
    label = pd.read_csv('../data/full_pos2.txt', sep = ' ')
    dataset_m1 = torch.tensor(np.array(dataset_m1))
    edge =  np.array(label).tolist()
    weight =  np.array(weight).tolist()

    #label_real = label['label']
    edge_dict = defaultdict(list)
    weight_dict = defaultdict(list)
    for edg in weight:
        edge_dict[int(edg[0])].append([int(edg[1]), int(edg[2])])
        weight_dict[int(edg[0])].append(edg[-1])


                
    """GCN"""
    class GCNNet(torch.nn.Module):
        def __init__(self):
            super(GCNNet, self).__init__()
            self.conv1 = GCNConv(572, 400)   
            self.conv2 = GCNConv(400, 30)    
            self.linear = nn.Linear(60,1)   


        def forward(self, data,node_pair):
            x, edge_index ,edge_weight= data.x, data.edge_index, data.edge_weight

            x = x.float()


            # x = self.conv1(x, edge_index)
            x = self.conv1(x, edge_index,edge_weight=edge_weight)# If you want to remove the edge weight element, replace it with the upper line
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            # x = self.conv2(x, edge_index)
            x = self.conv2(x,edge_index, edge_weight = edge_weight)# If you want to remove the edge weight element, replace it with the upper line
            drug_emb = x 

            x = torch.cat((x[node_pair[0]],x[node_pair[1]]), dim =1)
            x = self.linear(x)

            return x, drug_emb

    """GraphSAGE"""
    class EdgeSAGENet(torch.nn.Module):
        def __init__(self):
            super(EdgeSAGENet, self).__init__()
            self.conv1 = SAGEConv(572, 400)   
            self.conv2 = SAGEConv(400, 30)    
            # self.edge_linear = nn.Linear(60, 60)
            self.linear = nn.Linear(60,1)   


        def forward(self, data,node_pair):
            x, edge_index = data.x, data.edge_index
            x = x.float()

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x,edge_index)
            drug_emb = x 

            x = torch.cat((x[node_pair[0]],x[node_pair[1]]), dim =1)
            x = self.linear(x)

            return x, drug_emb
        

    """GAT"""
    class GATNet(torch.nn.Module):
        def __init__(self):
            super(GATNet, self).__init__()
            self.conv1 = GATConv(572, 400, heads=4, concat=True)   
            self.conv2 = GATConv(1600, 30, heads=4, concat=False)    
            self.linear = nn.Linear(60,1)   


        def forward(self, data,node_pair):
            x, edge_index = data.x, data.edge_index
            x = x.float()

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x,edge_index)
            drug_emb = x 

            x = torch.cat((x[node_pair[0]],x[node_pair[1]]), dim =1)
            x = self.linear(x)

            return x, drug_emb
        
    """GIN"""
    class GINNet(torch.nn.Module):
        def __init__(self):
            super(GINNet, self).__init__()
            nn1 = Sequential(Linear(572, 400), ReLU(), Linear(400, 400))
            self.conv1 = GINConv(nn1)
            nn2 = Sequential(Linear(400, 30), ReLU(), Linear(30, 30))
            self.conv2 = GINConv(nn2)
            self.linear = nn.Linear(60, 1)

        def forward(self, data, node_pair):
            x, edge_index = data.x, data.edge_index
            x = x.float()

            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            drug_emb = x

            x = torch.cat((x[node_pair[0]], x[node_pair[1]]), dim=1)
            x = self.linear(x)

            return x, drug_emb



    def embedding_data_generate(edge_dict,weight_dict,key): # edge embedding
        
        edge = edge_dict[key] 
        weight = weight_dict[key]
        
        label = torch.ones(len(edge)) 
        edge = torch.tensor(edge)
        X_train, X_test,w_train,w_test, y_train, y_test = train_test_split(edge, weight,label, test_size= 0.15,random_state=42) # train:val:test = 0.7:0.15:0.15
        X_train, X_val,w_train,w_val, y_train, y_val = train_test_split(X_train,w_train, y_train, test_size= 0.15,random_state=42) 
        neg_train = negative_sampling(X_train.t().contiguous(),572,len(X_train))
        t_node_pair = torch.cat([X_train.t().contiguous(),neg_train],dim =1) 
   
        t_label_n = torch.zeros(len(neg_train[0]))
        y_train = torch.cat([y_train,t_label_n],dim=0) 
    
        
        neg_val = negative_sampling(X_val.t().contiguous(),572,len(X_val))
        v_node_pair = torch.cat([X_val.t().contiguous(),neg_val],dim =1)
        v_label_n = torch.zeros(len(neg_val[0]))
        y_val = torch.cat([y_val,v_label_n],dim=0)
        

        neg_test = negative_sampling(X_test.t().contiguous(),572,len(X_test))
        Test_node_pair = torch.cat([X_test.t().contiguous(),neg_test],dim =1)
        Test_node_pair_out = X_test.t().contiguous() 
        t_label_n = torch.zeros(len(neg_test[0]))
        y_test = torch.cat([y_test,t_label_n],dim=0)

        
    
        train_graph = Data(x=dataset_m1, edge_index = X_train.t().contiguous(), label = y_train,edge_weight=torch.tensor(w_train)) 
    
        val_graph = Data(x=dataset_m1, edge_index = X_val.t().contiguous(), label = y_val,edge_weight=torch.tensor(w_val))
        test_graph = Data(x=dataset_m1, edge_index = X_test.t().contiguous(), label = y_test,edge_weight=torch.tensor(w_test))
        
       
        
        return train_graph, val_graph, test_graph, t_node_pair,v_node_pair, Test_node_pair, Test_node_pair_out



    def train_val(train_graph, val_graph,t_node_pair,v_node_pair, model):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        train_graph = train_graph.to(device)
        val_graph = val_graph.to(device)
        

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

        f1_list = []
        for epoch in range(120):
            model.train()
            
            optimizer.zero_grad()
            out,_ = model(train_graph,t_node_pair) # data, node_pair
            out= out.squeeze(1)
            loss = F.binary_cross_entropy(torch.sigmoid(out), train_graph.label)
            loss.backward()
            optimizer.step()
            

            model.eval()

            
            with torch.no_grad():
                
                out,drug_emb = model(train_graph,v_node_pair)
                out = out.squeeze(1)
                loss_1 = F.binary_cross_entropy(torch.sigmoid(out), val_graph.label)
                # print(loss_1)
                
                
                out[out>=0.5] = 1
                out[out<0.5] = 0
                

                acc_score_r = accuracy_score(out.cpu(),val_graph.label.cpu())
                f1_list.append(acc_score_r)
                
        
            
        # print(max(f1_list))
        return model,drug_emb

            
            
            
    models = {"GCN": GCNNet(),"GraphSAGE": EdgeSAGENet(), "GAT": GATNet(), "GIN": GINNet()}
    
    
    test = []
    for model_name, model in models.items():
        print('-------------------------------------')
        print(f'generate emb with {model_name} model')
        print('-------------------------------------')
        for key in tqdm(edge_dict.keys()):
   
            
            train_graph, val_graph, test_graph, t_node_pair,v_node_pair, Test_node_pair,Test_node_pair_out = embedding_data_generate(edge_dict,weight_dict,key)    
            
            model,drug_emb = train_val(train_graph, val_graph,t_node_pair,v_node_pair,model) 

            drug_emb = drug_emb.detach().cpu().numpy()
            df = pd.DataFrame(drug_emb)
    
            Test_node_pairs = Test_node_pair_out.t().contiguous().tolist()

            
            for j in range(len(Test_node_pairs)):
                
                test.append(Test_node_pairs[j])
            df.to_csv(f'../node_emb_v2/emb_{model_name}_m{i}_{key}.csv', index=False)
    
test = np.array(test)
test = pd.DataFrame(test)
test.to_csv(f'../data/test.csv', index=False)




