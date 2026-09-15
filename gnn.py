import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.nn import GCNConv


# ===============================
# LOAD DATA
# ===============================
data = pd.read_csv("train_clean.csv")

data = data.loc[:, data.nunique() > 2]


data["RUL"] = data["RUL"].clip(upper=125)

features=[c for c in data.columns if c not in ["engine_id","cycle","RUL"]]


# ===============================
# NORMALIZE
# ===============================
scaler=StandardScaler()
data[features]=scaler.fit_transform(data[features])


# ===============================
# SLIDING WINDOW
# ===============================
WINDOW=30
X=[]
y=[]

for engine in data.engine_id.unique():

    df=data[data.engine_id==engine]

    sensors=df[features].values
    rul=df["RUL"].values

    for i in range(len(df)-WINDOW):
        X.append(sensors[i:i+WINDOW])
        y.append(rul[i+WINDOW])

X=np.array(X)
y=np.array(y)

split=int(len(X)*0.8)
X_train,X_val=X[:split],X[split:]
y_train,y_val=y[:split],y[split:]


# ===============================
# BUILD SENSOR GRAPH
# ===============================
corr=data[features].corr().values

edges=[]
for i in range(len(features)):
    for j in range(len(features)):
        if abs(corr[i,j])>0.3 and i!=j:
            edges.append([i,j])

edge_index=torch.tensor(edges).t().contiguous()


# ===============================
# DEFINE REAL GNN MODEL
# ===============================
class GNN(nn.Module):
    def __init__(self,n_nodes):
        super().__init__()
        self.conv1 = GCNConv(n_nodes,128)
        self.conv2 = GCNConv(128,64)
        self.conv3 = GCNConv(64,32)
        self.fc = nn.Linear(32,1)

    def forward(self,x,edge_index):
        x = torch.relu(self.conv1(x,edge_index))
        x = torch.relu(self.conv2(x,edge_index))
        x = torch.relu(self.conv3(x,edge_index))
        x = torch.mean(x,dim=0)
        return self.fc(x)


model=GNN(WINDOW)

optimizer=torch.optim.Adam(model.parameters(),lr=0.005)
loss_fn=nn.MSELoss()


# ===============================
# TRAIN LOOP
# ===============================
for epoch in range(25):

    total=0

    for seq,target in zip(X_train,y_train):

      
        seq=torch.tensor(seq.T,dtype=torch.float32)
        target=torch.tensor([target],dtype=torch.float32)

        optimizer.zero_grad()
        pred=model(seq,edge_index)
        loss=loss_fn(pred,target)

        loss.backward()
        optimizer.step()

        total+=loss.item()

    print("epoch",epoch,"loss",total)


# ===============================
# VALIDATION
# ===============================
preds=[]

for seq in X_val:
    seq=torch.tensor(seq.T,dtype=torch.float32)
    preds.append(model(seq,edge_index).item())

mae=mean_absolute_error(y_val,preds)
rmse=np.sqrt(mean_squared_error(y_val,preds))

print("\nGNN MAE:",round(mae,2))
print("GNN RMSE:",round(rmse,2))


# SAVE MODEL
torch.save(model.state_dict(),"model_gnn.pt")
print("\nSaved model_gnn.pt")
