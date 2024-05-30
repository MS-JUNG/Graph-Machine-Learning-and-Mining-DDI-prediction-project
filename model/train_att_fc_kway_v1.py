"""
written by Sang Won Kim
for the course-work project of lecture "Graph Machine Learning and Mining"
Embedding and some code snippets used in this work are produced by MinSu Jung, YiJin Kim

This file is for the classifier with 
 - Self-Attention attached MLP,
 - Self-Attention attached K-Way Edge Predictor
for
 - node_emb_v1
 
usage 
python train_att_fc_kway_v1.py MLP > result_att_MLP_v1.out
python train_att_fc_kway_v1.py kway > result_att_kway_v1.out

"""
from copy import deepcopy
import torch
import torch.nn as nn
import pandas as pd 
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random
import torch.optim as optim
from utils.utils import *
from sklearn.model_selection import train_test_split
import sys
import csv
import torch.backends.cudnn as cudnn

seed_num = 42
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(seed_num)

numEdgeTypes = 65
dim = 256
numNodes = 572
batch_size = 128
attn_dim = 1024

def get_emb(f_num):
    data = pd.read_csv(f'../node_emb_v1/emb_m{f_num}.csv')
    data = torch.tensor(np.array(data), dtype=torch.float32)
    data_m = data 
    
    return data_m


def create_data(path, is_train):
    if is_train:
        data = load_training_data(path)
    else:
        data, _ = load_testing_data(path)

    del(data['label'])
    data = {int(key): [(int(pair[0]), int(pair[1])) for pair in value] 
               for key, value in data.items()}   # dict {0 : [(541, 280), (541, 43) ... ] }
    data_ = []
    for key,value in data.items():
        for edge in value:
            data_.append([key, edge[0], edge[1]])
    data_ = pd.DataFrame(data_)
    data = data_
                
    return data

class NodeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = attn_dim
        self.num_feats = dim 
        self.W_q = nn.Linear(self.num_feats, self.dim, dtype=torch.float32)
        self.W_k = nn.Linear(self.num_feats, self.dim, dtype=torch.float32)
        self.W_v = nn.Linear(self.num_feats, self.dim, dtype=torch.float32)
        
    def forward(self, embedding):
        Q = self.W_q(embedding)
        K = self.W_k(embedding)
        V = self.W_v(embedding)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores /= torch.sqrt(torch.tensor(self.num_feats, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        
        output = torch.matmul(attn_weights, V)
        
        return output # context vectors of node embeddings 
        

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        self.dropout = nn.Dropout(0.5)    
    
    def forward(self, x):
        x = self.dropout(self.bn1(nn.ReLU()(self.fc1(x))))
        x = self.dropout(self.bn2(nn.ReLU()(self.fc2(x))))
        x = self.fc3(x)
        
        return x
    
class KWayPred(nn.Module):
    
    def __init__(self, dim):
        super(KWayPred, self).__init__()
        self.dim = dim
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.randn(self.dim, self.dim, dtype=torch.float32)).cuda() for _ in range(numEdgeTypes)
            ]
        )
        
    def forward(self, h_u, h_v): # 각 batch_size x attn_dim
        output = torch.zeros(h_u.shape[0], numEdgeTypes).cuda()
        for i in range(numEdgeTypes):
            val = torch.mm(torch.mm(h_u, self.weights[i]), h_v.T).requires_grad_(True)
            output[:,i] = torch.diag(val, 0)

        return output

class EdgePredLayer(nn.Module):
    def __init__(self, embedding):
        super(EdgePredLayer, self).__init__()
        self.embedding = embedding.to(device) # 572 x 1140
        self.attn = NodeSelfAttention()
        if is_mlp:
            self.classifier = MLP(attn_dim * 2, [1000, 200, 100], numEdgeTypes).to(device)
        else:
            self.classifier = KWayPred(attn_dim).to(device)

    def forward(self, data): # attn_dim=256 here
        attn_emb = self.attn(self.embedding) # 572 x 256 
        start_emb = attn_emb[data[:, 0]].cuda() # batch_size x 256
        end_emb = attn_emb[data[:, 1]].cuda() # batch_size x 256
        if is_mlp:
            emb_cat = torch.concat([start_emb, end_emb], dim=1).cuda() # batch_size x (256x2) 
            output = self.classifier(emb_cat)
        else:
            output = self.classifier(start_emb, end_emb)
            
        return output

class EdgePred(nn.Module):
    
    def __init__(self, using_feats, embeddings):
        super(EdgePred, self).__init__()
        self.using_feats = using_feats
        self.embeddings = embeddings
        self.num_feats = dim
        
        self.feat_heads = nn.ModuleList([
            EdgePredLayer(embeddings[i]).to(device) for i in range(len(using_feats))
        ])
        
    def forward(self, data):
        data = data.to(device)
        output_accumulated = torch.zeros(numEdgeTypes, data.shape[0], requires_grad=True).t().cuda()
        # print(output_accumulated.shape)
        for i in range(len(self.using_feats)):
            output = self.feat_heads[i](data).cuda() # logit for each features 1, 2, 3, 4
            # print(output.shape)
            output_accumulated = output_accumulated + output
        
        return output_accumulated # logits accumulated 
        
class CustomDataset(Dataset):
    """
    Dataset class에서는 데이터의 node 번호와 label만 return하도록 하고, 모델 내에서 feature에 따른 embedding을 찾아서 계산하도록 구현
    """
    def __init__(self, data):
        self.item = np.array(data.iloc[:, 1:])
        self.label = np.array(data.iloc[:, 0])
                        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.item[idx], self.label[idx] 
        
def train_kway(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data, labels in tqdm(loader):
        labels = labels.type(torch.LongTensor).cuda()
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data).cuda()
        # outputs = outputs.t().cuda()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)   


def validate(model, loader, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data, labels in loader:
            labels = labels.type(torch.LongTensor).cuda()  
            data = data.to(device)
            outputs = model(data).t().cuda()
            _, predicted = torch.max(outputs.data, 0)
            # print("predicted shape : ", predicted.shape)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            outputs = outputs.t().cuda()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader), 100* correct / total

def test(model, loader):
    print("start testing")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            labels = labels.type(torch.LongTensor).cuda()  
            data = data.to(device)
            outputs = model(data).t().cuda()
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            outputs = outputs.t().cuda()
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

    #### Main        
if __name__ == "__main__":
    ### 1. parsing node embeddings 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_mlp = True if sys.argv[1] == 'MLP' else False
    attn_dim = 1024 if is_mlp else 256
    using_feats = [1, 2, 3, 4]
    
    embeddings = []
    for f_num in range(len(using_feats)):
        embeddings.append(get_emb(using_feats[f_num]))
    
    # print("embedding shape for using feat", using_feats[0], " : ", embeddings[0].shape, "dtype:", embeddings[0].dtype )
    # 572 x 30 
    
    ### 2. data parsing, dataset, dataloader

    train_d = create_data("../data/full_pos2.txt", is_train=True)
    csv_file_path = "../data/test.csv"
    test_l = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            test_l.append(row)
    
    train_d_ = pd.DataFrame()
    test_d = pd.DataFrame()
    for index, row in train_d.iterrows():
        edge = [str(row[1]), str(row[2])] 
        if edge in test_l:
            test_d = pd.concat([test_d, pd.DataFrame(row).transpose()])
        else:
            train_d_ = pd.concat([train_d_, pd.DataFrame(row).transpose()])
                
    
    # Else, use train_test_split to construct val set 
    train_d_, val_d = train_test_split(train_d_, test_size = 0.15)
    train_d = train_d_
    
    print("length of train dataset:", len(train_d)) # dataframe : [label, node1, node2] x len(train_d)  형태의 pd 데이터프레임 
    print("length of val dataset:", len(val_d))
    print("length of test dataset:", len(test_d))
    
    train_dataset = CustomDataset(train_d)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True) 
    val_dataset = CustomDataset(val_d)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CustomDataset(test_d)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    ### 3. model instance generation
    
    model = EdgePred(using_feats, embeddings)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)#, momentum=0.9)
    
    
    ### 4. training    
    epochs = 30
    val_loss_max= 100
    best_val_loss = float('inf')
    best_model_wts = None
    print("start train")
    
    for epoch in range(epochs):
        train_loss = train_kway(model, train_dataloader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_dataloader, criterion)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc}')
        
        if val_loss < best_val_loss:
            print("model update")
            best_val_loss = val_loss
            
            best_model_wts = deepcopy(model.state_dict())

    torch.save(best_model_wts, 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    
    new_model = EdgePred(using_feats, embeddings)
    new_model.load_state_dict(torch.load('best_model.pth'))


    accuracy = test(new_model, test_dataloader)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print("used features : ", using_feats)
else :
    print("kway, mlp imported")
