"""
written by Sang Won Kim
for the course-work project of lecture "Graph Machine Learning and Mining"
Embedding and some code snippets used in this work are produced by MinSu Jung, YiJin Kim

This file is for the classifier with 
 - K-Way Edge Predictor
for
 - node_emb_v1
 
usage
python train_kway_v1.py > result_kway_v1.out
"""

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
from copy import deepcopy

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
batch_size = 256

def get_emb(f_num):
    data = pd.read_csv(f'../node_emb_v1/emb_m{f_num}.csv')
    data = torch.tensor(np.array(data))
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

class kWayLayer(nn.Module):
    def __init__(self, using_feats, using_feat):
        super().__init__()
        self.using_feat = using_feat
        self.using_feats = using_feats
        self.num_feats = dim
        self.weights = nn.ParameterList(
            [nn.Parameter(
                torch.randn(self.num_feats, self.num_feats, dtype=torch.float64)).cuda()
            for _ in range(numEdgeTypes)
            ]
        )
        
    def forward(self, data):
        # start_emb : h_u
        # end_emb : h_v
        start_emb = torch.Tensor(data[0][self.using_feats.index(self.using_feat)]).cuda()
        end_emb = torch.Tensor(data[1][self.using_feats.index(self.using_feat)]).cuda()
        output = torch.zeros(numEdgeTypes, start_emb.shape[0], requires_grad=True).cuda()
        for n in range(numEdgeTypes):
            val = torch.mm(torch.mm(start_emb, self.weights[n]), end_emb.T).requires_grad_(True)
            output[n] = output[n] + torch.diag(val, 0)
        
        return output

class kWayEdgePred(nn.Module):
    
    def __init__(self, using_feats):
        super().__init__()
        self.using_feats = using_feats
        self.num_feats = dim
        self.feat_heads = nn.ModuleList([
            kWayLayer(using_feats, using_feat) for using_feat in using_feats
        ])
    
    def forward(self, data):
        if data[0][0].shape[0] != batch_size:
            output_accumulated = torch.zeros(numEdgeTypes, data[0][0].shape[0], requires_grad=True).cuda()
        else:
            output_accumulated = torch.zeros(numEdgeTypes, batch_size, requires_grad=True).cuda()
        
        for using_feat in self.using_feats:
            output = self.feat_heads[self.using_feats.index(using_feat)](data)
            output_accumulated = output_accumulated + output
        
        return output_accumulated
        
        
class CustomDataset(Dataset):
    def __init__(self, data, using_feats, embeddings):
        self.item = np.array(data.iloc[:, 1:])
        self.label = np.array(data.iloc[:, 0])
        self.using_feats = using_feats
        self.embeddings = embeddings
                        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        """  
        emb_by_feats : [[0번 데이터의 start node의 embeddings], [0번 데이터의 end node의 embeddings]]
        각 embeddings은 사용하는 feature 가지 수에 따라 1~4개의 embedding이 들어감. 
        label : 0번 data의 edge type
        [[], []], label
        """
        start = self.item[idx][0]
        embs_start = [self.embeddings[i][start] for i in range(len(self.using_feats))]
        
        end = self.item[idx][1]
        embs_end = [self.embeddings[i][end] for i in range(len(self.using_feats))]
        
        embs_by_feats = [embs_start, embs_end]
        
        label = self.label[idx]

        return embs_by_feats, label
    
def train_kway(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    i = 0 # flag
    for data, labels in tqdm(loader):
        labels = labels.type(torch.LongTensor).cuda()
        optimizer.zero_grad()
        outputs = model(data)
        outputs = outputs.t().cuda()

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
    i = 0  # flag for examplar testing
    with torch.no_grad():
        for data, labels in loader:
            labels = labels.type(torch.LongTensor).cuda()  
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            outputs = outputs.t().cuda()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader), 100* correct / total

def test(models, loader):
    print("start testing")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in loader:
            # data, labels = data.cuda(),labels.cuda()
            labels = labels.type(torch.LongTensor).cuda()  
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
    
    #### Main        
if __name__ == "__main__":
    # GCN, GAT, GraphSAGE, GIN 
    # usage : CUDA_VISIBLE_DEVICES=3 python train_kway.py > result_kway.out
    # 

    ### 1. parsing node embeddings 
    using_feats = [1, 2, 3, 4]
    
    embeddings = []
    for f_num in range(len(using_feats)):
        embeddings.append(get_emb(using_feats[f_num]))
    # print("embedding shape for using feat", using_feats[0], " : ", embeddings[0].shape)
    # 572 x 30
    
    
    ### 2. model instance generation
    model = kWayEdgePred(using_feats)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    
    ### 3. data parsing, dataset, dataloader
        # train data 
    train_d = create_data("../data/full_pos2.txt", is_train=True)

    # If edge in test.csv, put it into test_d
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
    
    print("length of train dataset:", len(train_d))
    print("length of val dataset:", len(val_d))
    print("length of test dataset:", len(test_d))

    
    train_dataset = CustomDataset(train_d, using_feats, embeddings)
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True) 
    val_dataset = CustomDataset(val_d, using_feats, embeddings)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = CustomDataset(test_d, using_feats, embeddings)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    
    
    ### 4. training    
    best_val_loss = float('inf')
    epochs = 30
    val_loss_max= 100
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
    
    
    new_model = kWayEdgePred(using_feats)
    new_model.load_state_dict(torch.load('best_model.pth'))

    accuracy = test(new_model, test_dataloader)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print("used features : ", using_feats)
else :
    print("train_kway imported")
