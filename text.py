import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn


# =========================
# LOAD DATA
# =========================
data = pd.read_csv("train_clean.csv")


data = data.loc[:, data.nunique() > 2]

data["RUL"] = data["RUL"].clip(upper=125)

features=[c for c in data.columns if c not in ["engine_id","cycle","RUL"]]


# =========================
# NORMALIZE FOR TEXT LOGIC
# =========================
scaler=StandardScaler()
data[features]=scaler.fit_transform(data[features])


# =========================
# CREATE SENSOR TEXT
# =========================
def make_text(row):

    desc=[]

    for i,f in enumerate(features[:8]):   
        val=row[f]

        if val>1:
            desc.append(f"{f} high")
        elif val<-1:
            desc.append(f"{f} low")
        else:
            desc.append(f"{f} normal")

    return " ".join(desc)


data["text"]=data.apply(make_text,axis=1)


# =========================
# LOAD TRANSFORMER
# =========================
model_embed=SentenceTransformer('all-MiniLM-L6-v2')

texts=data["text"].tolist()

print("Encoding text ...")
X=model_embed.encode(texts,show_progress_bar=True)

y=data["RUL"].values


# =========================
# TRAIN / VAL SPLIT
# =========================
split=int(len(X)*0.8)

X_train,X_val=X[:split],X[split:]
y_train,y_val=y[:split],y[split:]


# =========================
# SIMPLE TEXT PREDICTOR NN
# =========================
class TextModel(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc1=nn.Linear(dim,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,1)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)


model=TextModel(X.shape[1])

opt=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn=nn.MSELoss()


# =========================
# TRAIN
# =========================
for epoch in range(12):

    total=0

    for xi,yi in zip(X_train,y_train):

        xi=torch.tensor(xi,dtype=torch.float32)
        yi=torch.tensor([yi],dtype=torch.float32)

        opt.zero_grad()
        pred=model(xi)
        loss=loss_fn(pred,yi)
        loss.backward()
        opt.step()

        total+=loss.item()

    print("epoch",epoch,"loss",total)


# =========================
# VALIDATE
# =========================
preds=[]

for xi in X_val:
    xi=torch.tensor(xi,dtype=torch.float32)
    preds.append(model(xi).item())

mae=mean_absolute_error(y_val,preds)
rmse=np.sqrt(mean_squared_error(y_val,preds))

print("\nTEXT MODEL MAE:",round(mae,2))
print("TEXT MODEL RMSE:",round(rmse,2))


# SAVE
torch.save(model.state_dict(),"model_text.pt")
print("\nSaved model_text.pt")
4
