import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader


# =============================
# LOAD DATA
# =============================
data = pd.read_csv("train_clean.csv")
data = data.loc[:, data.nunique()>2]
data["RUL"] = data["RUL"].clip(upper=125)

features=[c for c in data.columns if c not in ["engine_id","cycle","RUL"]]

scaler=StandardScaler()
data[features]=scaler.fit_transform(data[features])


# =============================
# TEXT DESCRIPTION
# =============================
def make_text(row):
    desc=[]
    for f in features[:8]:
        v=row[f]
        if v>1: desc.append(f"{f} high")
        elif v<-1: desc.append(f"{f} low")
        else: desc.append(f"{f} normal")
    return " ".join(desc)

data["text"]=data.apply(make_text,axis=1)

embed_model=SentenceTransformer("all-MiniLM-L6-v2")
text_vec=embed_model.encode(data["text"].tolist())


# =============================
# SLIDING WINDOW
# =============================
WINDOW=30

Xs=[]
Xt=[]
y=[]

idx=0

for engine in data.engine_id.unique():

    df=data[data.engine_id==engine]

    sensors=df[features].values
    rul=df["RUL"].values
    texts=text_vec[idx:idx+len(df)]
    idx+=len(df)

    for i in range(len(df)-WINDOW):

        Xs.append(sensors[i:i+WINDOW])     
        Xt.append(texts[i+WINDOW])
        y.append(rul[i+WINDOW])

Xs=np.array(Xs,dtype=np.float32)
Xt=np.array(Xt,dtype=np.float32)
y=np.array(y,dtype=np.float32)


# =============================
# BUILD SENSOR GRAPH (REAL)
# =============================
corr=data[features].corr().values

edges=[]
for i in range(len(features)):
    for j in range(len(features)):
        if abs(corr[i,j])>0.3 and i!=j:
            edges.append([i,j])

edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous()


# =============================
# DATASET
# =============================
class EngineDataset(Dataset):
    def __init__(self,Xs,Xt,y):
        self.Xs=Xs
        self.Xt=Xt
        self.y=y

    def __len__(self): return len(self.Xs)

    def __getitem__(self,i):
        return (
            torch.tensor(self.Xs[i]),
            torch.tensor(self.Xt[i]),
            torch.tensor(self.y[i])
        )


split=int(len(Xs)*0.8)

train_loader=DataLoader(
    EngineDataset(Xs[:split],Xt[:split],y[:split]),
    batch_size=64,shuffle=True)

val_loader=DataLoader(
    EngineDataset(Xs[split:],Xt[split:],y[split:]),
    batch_size=128)


# =============================
# REAL HYBRID MODEL
# =============================
class Hybrid(nn.Module):

    def __init__(self,text_dim,num_sensors):
        super().__init__()

        # -------- LSTM (temporal) --------
        self.lstm=nn.LSTM(num_sensors,64,batch_first=True)

        # -------- REAL GNN --------
        # each sensor = node
        # node feature = WINDOW length

        self.conv1=GCNConv(WINDOW,64)
        self.conv2=GCNConv(64,32)

        # -------- TEXT --------
        self.text_fc=nn.Linear(text_dim,64)

        # -------- FINAL --------
        self.final=nn.Sequential(
            nn.Linear(64+32+64,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,sensor,text):

        # sensor shape: [B,30,S]

        # ----- LSTM -----
        _,(h,_) = self.lstm(sensor)
        sensor_feat=h[-1]

        # ----- TRUE GNN -----
        B,T,S=sensor.shape

        g_list=[]

        for i in range(B):

            # nodes = sensors
            # features = 30 time values
            node_feat=sensor[i].transpose(0,1)  
            g=torch.relu(self.conv1(node_feat,edge_index))
            g=torch.relu(self.conv2(g,edge_index))

            g=g.mean(dim=0)
            g_list.append(g)

        graph_feat=torch.stack(g_list)

        # ----- TEXT -----
        text_feat=torch.relu(self.text_fc(text))

        # ----- FUSION -----
        x=torch.cat([sensor_feat,graph_feat,text_feat],dim=1)

        return self.final(x).squeeze()


model=Hybrid(Xt.shape[1],len(features))

opt=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()


# =============================
# IMPROVED TRAINING 
# =============================

model = Hybrid(Xt.shape[1], len(features))

opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)


loss_fn = nn.SmoothL1Loss(beta=10)   

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

best = 999

for epoch in range(50):

    model.train()

    for s, t, r in train_loader:

        opt.zero_grad()

        pred = model(s, t)

        # -------- RUL weighting (reduces high-RUL variance) --------
        weights = 1 + (r / 125)   
        loss = loss_fn(pred, r)
        loss = (loss * weights).mean()

        loss.backward()

        # -------- Gradient clipping (stability) --------
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

    scheduler.step()

    # ================= VALIDATION =================
    model.eval()

    preds = []
    true = []

    with torch.no_grad():
        for s, t, r in val_loader:
            p = model(s, t)
            preds += p.tolist()
            true += r.tolist()

    mae = mean_absolute_error(true, preds)
    rmse = np.sqrt(mean_squared_error(true, preds))

    print(f"Epoch {epoch} | MAE: {round(mae,2)} | RMSE: {round(rmse,2)}")

    if mae < best:
        best = mae
        torch.save(model.state_dict(), "REAL_GNN.pt")
        print("Saved BEST")

print("BEST MAE:", best)
