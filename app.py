# =========================================================
# IMPORTS
# =========================================================
from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv
from groq import Groq
import os
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

# =========================================================
# INITIALIZE FLASK
# =========================================================
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

WINDOW = 30

# =========================================================
# DATABASE
# =========================================================
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# =========================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

def explain(rul):
    if not client:
        return "AI explanation unavailable. Please set GROQ_API_KEY environment variable."

    if rul <= 25:
        level = "High Risk"
    elif rul <= 60:
        level = "Moderate Risk"
    else:
        level = "Low Risk"

    prompt = f"""
    The aircraft engine has a predicted Remaining Useful Life (RUL) of {rul} cycles.
    Risk Level: {level}

    Generate a technical maintenance summary in clear bullet points.

    Requirements:
    - Use 6 to 8 detailed technical bullet points
    - Each point must be 1–2 lines
    - Mention the RUL value naturally
    - Include sensor degradation reasoning (temperature, vibration, pressure trends)
    - Include mechanical wear explanation
    - Include operational risk impact
    - Include maintenance planning strategy
    - Do NOT use headings
    - Do NOT use placeholders
    - Do NOT write as paragraph
    - Keep it professional and engineering-focused
    """

    try:
        r = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    except:
        return "AI explanation unavailable."
# =========================================================
# LOAD TRAIN ARTIFACTS
# =========================================================
def load_train_artifacts():
    train = pd.read_csv("train_clean.csv")
    train = train.loc[:, train.nunique() > 2]
    train["RUL"] = train["RUL"].clip(upper=125)

    features = [c for c in train.columns if c not in ["engine_id","cycle","RUL"]]

    scaler = StandardScaler()
    scaler.fit(train[features])

    corr = train[features].corr().values

    edges = []
    for i in range(len(features)):
        for j in range(len(features)):
            if abs(corr[i,j]) > 0.3 and i != j:
                edges.append([i,j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return scaler, features, edge_index

scaler, features, edge_index = load_train_artifacts()

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================
embed = SentenceTransformer("all-MiniLM-L6-v2")

# =========================================================
# HYBRID MODEL
# =========================================================
class Hybrid(nn.Module):

    def __init__(self,text_dim,num_sensors):
        super().__init__()

        self.lstm=nn.LSTM(num_sensors,64,batch_first=True)
        self.conv1=GCNConv(WINDOW,64)
        self.conv2=GCNConv(64,32)
        self.text_fc=nn.Linear(text_dim,64)

        self.final=nn.Sequential(
            nn.Linear(64+32+64,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,sensor,text):

        sensor=torch.tensor(sensor,dtype=torch.float32).unsqueeze(0)

        # LSTM
        _,(h,_) = self.lstm(sensor)
        sensor_feat=h[-1]

        # GNN
        node_feat=sensor[0].transpose(0,1)
        g=torch.relu(self.conv1(node_feat,edge_index))
        g=torch.relu(self.conv2(g,edge_index))
        graph_feat=g.mean(dim=0,keepdim=True)

        # TEXT
        t=torch.tensor(text,dtype=torch.float32).unsqueeze(0)
        text_feat=torch.relu(self.text_fc(t))

        x=torch.cat([sensor_feat,graph_feat,text_feat],dim=1)

        return self.final(x).squeeze()

# =========================================================
# LOAD TRAINED MODEL
# =========================================================
model = Hybrid(384, len(features))
model.load_state_dict(torch.load("REAL_GNN.pt", map_location="cpu"))
model.eval()

# =========================================================
# ROUTES
# =========================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ================= REGISTER =================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        try:
            conn = get_db_connection()
            conn.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                         (name, email, password))
            conn.commit()
            conn.close()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already registered!", "danger")

    return render_template('register.html')

# ================= LOGIN =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['logged_in'] = True
            session['email'] = email
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials!", "danger")

    return render_template('login.html')

# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# ================= PREDICT =================
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if not session.get('logged_in'):
        return redirect(url_for('login'))

    cols=['engine_id','cycle','setting1','setting2','setting3']+[f's{i}' for i in range(1,22)]
    test=pd.read_csv("data/test_FD001.txt",sep="\s+",header=None)
    test.columns=cols

    true_rul=pd.read_csv("data/RUL_FD001.txt",header=None)

    engines = sorted(test.engine_id.unique())

    if request.method == 'POST':

        engine = int(request.form['engine'])
        df=test[test.engine_id==engine].copy()
        real_value=int(true_rul.iloc[engine-1,0])

        df[features]=scaler.transform(df[features])
        arr=df[features].values

        if len(arr)>=WINDOW:
            sensor=arr[-WINDOW:]
        else:
            pad=np.repeat(arr[0:1],WINDOW-len(arr),axis=0)
            sensor=np.vstack([pad,arr])

        def make_text(row):
            desc=[]
            for f in features[:8]:
                v=row[f]
                if v>1: desc.append(f"{f} high")
                elif v<-1: desc.append(f"{f} low")
                else: desc.append(f"{f} normal")
            return " ".join(desc)

        text_vec=embed.encode([make_text(df.iloc[-1])])[0]

        with torch.no_grad():
            rul=int(np.clip(model(sensor,text_vec).item(),0,125))
        # ================= ACCURACY CALCULATION =================
        if real_value > 0:
            accuracy = 100 - (abs(rul - real_value) / real_value * 100)
        else:
            accuracy = 0

        accuracy = round(accuracy, 1)


        if accuracy < 0:
            accuracy = 0
        # ================= RISK LEVEL LOGIC =================
        if rul <= 25:
            risk_level = "HIGH"
            risk_color = "red"
        elif rul <= 60:
            risk_level = "MODERATE "
            risk_color = "orange"
        else:
            risk_level = "LOW"
            risk_color = "green"

        explanation = explain(rul)

        # GRAPH 1
        fig, ax = plt.subplots()
        ax.bar(["Predicted","Actual"],[rul,real_value])
        ax.set_ylabel("RUL Cycles")
        img1 = BytesIO()
        plt.savefig(img1, format='png')
        img1.seek(0)
        graph1 = base64.b64encode(img1.getvalue()).decode()

        # GRAPH 2
        fig2, ax2 = plt.subplots()
        ax2.plot(sensor.mean(axis=1))
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Average Sensor Value")
        img2 = BytesIO()
        plt.savefig(img2, format='png')
        img2.seek(0)
        graph2 = base64.b64encode(img2.getvalue()).decode()

        # GRAPH 3
        models=["Random Forest","CNN","LSTM","Hybrid GNN"]
        mae=[15,12,19,10.3]
        fig3, ax3 = plt.subplots()
        ax3.bar(models,mae)
        ax3.set_ylabel("MAE")
        img3 = BytesIO()
        plt.savefig(img3, format='png')
        img3.seek(0)
        graph3 = base64.b64encode(img3.getvalue()).decode()

        return render_template('predict.html',
                               engines=engines,
                               selected_engine=engine,
                               rul=rul,
                               real_value=real_value,
                               accuracy=accuracy,
                               risk_level=risk_level,
                               risk_color=risk_color,
                               explanation=explanation,
                               graph1=graph1,
                               graph2=graph2,
                               graph3=graph3)

    return render_template('predict.html', engines=engines)
@app.route('/analysis')
def analysis():

    #  Restrict access
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('analysis.html')
# =========================================================
# MAIN
# =========================================================
if __name__ == '__main__':
    init_db()
    print(" Flask app running on http://127.0.0.1:5050/")
    app.run(debug=True, port=5050)