import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Flatten
from tensorflow.keras.models import save_model


# =====================================================
# LOAD DATA
# =====================================================
data = pd.read_csv("train_clean.csv")


data = data.loc[:, data.nunique() > 2]

data["RUL"] = data["RUL"].clip(upper=125)


features = [c for c in data.columns if c not in ["engine_id","cycle","RUL"]]


# =====================================================
# SCALE FEATURES
# =====================================================
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])


joblib.dump(scaler,"scaler.pkl")


# =====================================================
# CREATE SLIDING WINDOWS
# =====================================================
WINDOW = 30

X=[]
y=[]

for engine in data.engine_id.unique():

    df = data[data.engine_id==engine]

    sensors = df[features].values
    rul = df["RUL"].values

    for i in range(len(df)-WINDOW):
        X.append(sensors[i:i+WINDOW])
        y.append(rul[i+WINDOW])

X=np.array(X)
y=np.array(y)

print("Dataset:",X.shape)


# =====================================================
# TRAIN / VALID SPLIT
# =====================================================
split=int(len(X)*0.8)

X_train,X_val=X[:split],X[split:]
y_train,y_val=y[:split],y[split:]


# =====================================================
# METRIC FUNCTION
# =====================================================
def evaluate(name,true,pred):
    mae=mean_absolute_error(true,pred)
    rmse=np.sqrt(mean_squared_error(true,pred))
    print(name," MAE:",round(mae,2)," RMSE:",round(rmse,2))
    return mae,rmse


results=[]


# =====================================================
# RANDOM FOREST
# =====================================================
X_train_rf=X_train.reshape(X_train.shape[0],-1)
X_val_rf=X_val.reshape(X_val.shape[0],-1)

rf=RandomForestRegressor(n_estimators=150,n_jobs=-1)
rf.fit(X_train_rf,y_train)

pred=rf.predict(X_val_rf)
mae,rmse=evaluate("RandomForest",y_val,pred)
results.append(["RandomForest",mae,rmse])

joblib.dump(rf,"model_randomforest.pkl")


# =====================================================
# CNN
# =====================================================
cnn=Sequential([
    Conv1D(64,3,activation='relu',input_shape=(WINDOW,X.shape[2])),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(1)
])

cnn.compile(optimizer='adam',loss='mse')
cnn.fit(X_train,y_train,epochs=18,batch_size=128,verbose=1)

pred=cnn.predict(X_val).flatten()
mae,rmse=evaluate("CNN",y_val,pred)
results.append(["CNN",mae,rmse])

cnn.save("model_cnn.keras")


# =====================================================
# LSTM
# =====================================================
lstm=Sequential([
    LSTM(120,return_sequences=True,input_shape=(WINDOW,X.shape[2])),
    LSTM(60),
    Dense(1)
])

lstm.compile(optimizer='adam',loss='mse')
lstm.fit(X_train,y_train,epochs=18,batch_size=128)

pred=lstm.predict(X_val).flatten()
mae,rmse=evaluate("LSTM",y_val,pred)
results.append(["LSTM",mae,rmse])

lstm.save("model_lstm.keras")


# =====================================================
# THRESHOLD BASELINE
# =====================================================
mean=np.mean(X_train)
std=np.std(X_train)

threshold_pred=[20 if np.max(np.abs(x-mean))>2*std else 100 for x in X_val]

mae,rmse=evaluate("Threshold",y_val,threshold_pred)
results.append(["Threshold",mae,rmse])


# =====================================================
# SAVE RESULTS TABLE
# =====================================================
pd.DataFrame(results,columns=["Model","MAE","RMSE"]).to_csv(
    "baseline_results.csv",index=False
)

print("\nALL MODELS SAVED")
