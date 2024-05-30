import pandas as pd 
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

# Load node embeddings from CSV files
data_m1 = pd.read_csv(f'../node_emb_v1/emb_m1.csv')
data_m1 = torch.tensor(np.array(data_m1))
data_m2 = pd.read_csv(f'../node_emb_v1/emb_m2.csv')
data_m2 = torch.tensor(np.array(data_m2))
data_m3 = pd.read_csv(f'../node_emb_v1/emb_m3.csv')
data_m3 = torch.tensor(np.array(data_m3))
data_m4 = pd.read_csv(f'../node_emb_v1/emb_m4.csv')
data_m4 = torch.tensor(np.array(data_m4))

# Concatenate all embeddings into one tensor
data = [data_m1,data_m2,data_m3,data_m4]

data_total = torch.concat(data ,dim=1)

# Load edge labels and test edges
label = pd.read_csv('../data/full_pos2.txt', sep = ' ')
label = (np.array(label)).tolist()
test_edge = pd.read_csv('../data/test.csv')

label_train = []
label_test = []
data_edge_total =[]
data_edge_test_total =[]
test_edge = np.array(test_edge).tolist()

# Concat embedding of node to create edge embedding & split data into training and test sets
for i in tqdm(range(0,len(label))):
    if label[i][0] in range(0,65):

        if [label[i][1],label[i][2]] in test_edge:
            data_edge_test_total.append(np.array(torch.concat((data_total[label[i][1]].unsqueeze(0).clone().detach() , data_total[label[i][2]].unsqueeze(0).clone().detach()),dim=1)).tolist())            
            label_test.append(label[i][0])

        else:
            data_edge_total.append(np.array(torch.concat((data_total[label[i][1]].unsqueeze(0).clone().detach() , data_total[label[i][2]].unsqueeze(0).clone().detach()),dim=1)).tolist())
            label_train.append(label[i][0])
            
            pass

train_data = np.array(data_edge_total).squeeze(1)
test_data = np.array(data_edge_test_total).squeeze(1)
train_label = np.array(label_train)
test_label = np.array(label_test)

### Define the MLP model for edge classification
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

# Create an instance of the model
input_size = train_data.shape[1] 
hidden_size = [3000,1000,200,100]  
output_size = 65 
model = MLP(input_size, hidden_size, output_size)

### Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = torch.tensor(self.data[idx]).float()
        label = torch.tensor(self.labels[idx]).long() # Ensure labels are long for CrossEntropyLoss
        return item, label

X_train, X_val, y_train, y_val  = train_test_split(train_data,train_label, test_size = 0.2, random_state = 42)

# Create dataset and dataloader objects
tr_dataset = CustomDataset(X_train, y_train)
te_dataset = CustomDataset(test_data, test_label)
val_dataset = CustomDataset(X_val, y_val)

trdata_loader = DataLoader(tr_dataset, batch_size=64, shuffle=True)
tedata_loader = DataLoader(te_dataset, batch_size=64, shuffle=True)
valdata_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
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
    return total_loss / len(loader),model

# Validation function
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.cuda(),labels.cuda()
            labels = labels.type(torch.LongTensor).cuda()  
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        val_loss = total_loss / len(loader)
  
            
            
    return total_loss / len(loader)

# Testing function
def test(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.cuda(),labels.cuda()
            labels = labels.type(torch.LongTensor).cuda()  
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Train and validate the model
epochs = 30
best_val_loss = float('inf')
best_model_state = None
for epoch in range(epochs):
    train_loss,model = train(model, trdata_loader, optimizer, criterion)
    val_loss = validate(model, valdata_loader, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
    

# Load the best model and evaluate on the test set
model.load_state_dict(best_model_state)
accuracy = test(model, tedata_loader)
print(f'Best Test Accuracy: {accuracy:.2f}%')