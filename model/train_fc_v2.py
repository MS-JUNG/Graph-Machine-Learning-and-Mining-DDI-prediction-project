import pandas as pd 
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import wandb
import torch.backends.cudnn as cudnn
import random

seed_num = 42
print("seed number : {}".format(seed_num))

torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
np.random.seed(seed_num)
random.seed(seed_num)


def data1(mode):

    data_m1 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m1_0.csv')
    data_m1 = torch.tensor(np.array(data_m1))
    data_m2 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m2_0.csv')
    data_m2 = torch.tensor(np.array(data_m2))
    data_m3 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m3_0.csv')
    data_m3 = torch.tensor(np.array(data_m3))
    data_m4 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m4_0.csv')
    data_m4 = torch.tensor(np.array(data_m4))


    for i in range(1,65):
        
        data_1 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m1_{i}.csv')

        data_1 = torch.tensor(np.array(data_1))
        # data = data.reshape(data.shape[0],data.shape[1],1)

        data_m1 = torch.concat((data_m1,data_1),dim=1)
    for i in range(1,65):
        
        data_2 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m2_{i}.csv')

        data_2 = torch.tensor(np.array(data_2))


        data_m2 = torch.concat((data_m2,data_2),dim=1)

    for i in range(1,65):
        
        data_3 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m3_{i}.csv')

        data_3 = torch.tensor(np.array(data_3))

        data_m3 = torch.concat((data_m3,data_3),dim=1)
    for i in range(1,65):
        
        data_4 = pd.read_csv(f'../node_emb_v2/emb_{mode}_m4_{i}.csv')

        data_4 = torch.tensor(np.array(data_4))
     
        data_m4 = torch.concat((data_m4,data_4),dim=1)    

    label = pd.read_csv('../data/full_pos2.txt', sep = ' ')
    data = [data_m1,data_m2,data_m3,data_m4]
    data_total = torch.concat(data ,dim=1)
    label = (np.array(label)).tolist()
    test_edge = pd.read_csv('../data/test.csv')



    label_train = []
    label_test = []
    data_edge_total =[]
    data_edge_test_total =[]
    test_edge = np.array(test_edge).tolist()

    for i in tqdm(range(0,len(label))):
        if label[i][0] in range(0,65):

            if [label[i][1],label[i][2]] in test_edge:
                data_edge_test_total.append(np.array(torch.concat((data_total[label[i][1]].unsqueeze(0).clone().detach() , data_total[label[i][2]].unsqueeze(0).clone().detach()),dim=1)).tolist())
      
                
                label_test.append(label[i][0])

      
            else:
                data_edge_total.append(np.array(torch.concat((data_total[label[i][1]].unsqueeze(0).clone().detach() , data_total[label[i][2]].unsqueeze(0).clone().detach()),dim=1)).tolist())
                
                label_train.append(label[i][0])
                
                pass
    return data_edge_total, data_edge_test_total,label_train,label_test



models = ["GCN","GAT", "GraphSAGE", "GIN"]
# models = ["GCN"]


def process_model(model_name):
    
    data_edge_total,data_edge_test_total,label_train,label_test = data1(model_name)
    train_data = np.array(data_edge_total)
    test_data = np.array(data_edge_test_total)
    train_data = train_data.squeeze(1)
    test_data = test_data.squeeze(1)
    
    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    train_label = np.array(label_train)
    test_label = np.array(label_test)

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    # train_label = np.array(train_label).squeeze(axis=1)
    # test_label = np.array(test_label).squeeze(axis=1)

    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size[0]),
                nn.ReLU(),
                nn.Linear(hidden_size[0], hidden_size[1]),
                nn.ReLU(),
                nn.Linear(hidden_size[1], hidden_size[2]),
                nn.ReLU(),
                nn.Linear(hidden_size[2],hidden_size[3]),
                nn.ReLU(),
                nn.Linear(hidden_size[3], output_size)
                
            )

        def forward(self, x):
            return self.layers(x)

    
    input_size = 3900 * 4
    hidden_size = [3000,1000,200,100]  
    output_size = 65  

    model = MLP(input_size, hidden_size, output_size)


    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = torch.tensor(self.data[idx]).float()
            label = torch.tensor(self.labels[idx])
            return item, label



    
    X_train, X_val, y_train, y_val  = train_test_split(train_data,train_label, test_size = 0.2)

    tr_dataset = CustomDataset(X_train, y_train)
    te_dataset = CustomDataset(test_data, test_label)
    val_dataset = CustomDataset(X_val, y_val)

    trdata_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)
    tedata_loader = DataLoader(te_dataset, batch_size=64, shuffle=True)
    valdata_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    model = model.cuda()
    def train(model, loader, optimizer, criterion):
        model.train()
        total_loss = 0
        for data, labels in tqdm(loader):
            data, labels = data.cuda(),labels.cuda()
            labels = labels.type(torch.LongTensor).cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(loader)

            
        return total_loss / len(loader),model


    def validate(model, loader, criterion):
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        best_val_loss = 1000000000
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.cuda(),labels.cuda()
                labels = labels.type(torch.LongTensor).cuda()  
                outputs = model(data)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            v_acc = 100 * (correct / total)
            # wandb.log({f"{model_name}_val_acc" : v_acc})
            val_loss = total_loss / len(loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'./best_model/best_model_{model_name}.pth')

        return total_loss / len(loader)

    
    def test(model, loader):
        model.eval()
        correct = 0
        total = 0
        model.load_state_dict(torch.load(f'./best_model/best_model_{model_name}.pth'))
        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.cuda(),labels.cuda()
                labels = labels.type(torch.LongTensor).cuda()  
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_acc = 100 * (correct / total)
            # wandb.log({f"{model_name}_test_acc" : test_acc})
        return 100 * correct / total


    epochs = 30
    val_loss_max= 100
    for epoch in range(epochs):
        train_loss,model = train(model, trdata_loader, optimizer, criterion)
        val_loss = validate(model, valdata_loader, criterion)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        

    accuracy = test(model, tedata_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')



for model in models:
    print('----------------------------------')
    print(f"Result of {model} model")
    print('----------------------------------')
    process_model(model)

