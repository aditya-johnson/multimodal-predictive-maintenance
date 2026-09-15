# ✈️ Multimodal Industrial Predictive Maintenance using GNN + LLM Fusion

An end-to-end AI-powered predictive maintenance platform for commercial aircraft jet engines (built on the **NASA C-MAPSS dataset**). It combines **Time-Series Deep Learning (LSTM)**, **Graph Neural Networks (GNN)**, and **Large Language Models (Groq)** to accurately predict Remaining Useful Life (RUL) and generate automated, plain-English engineering maintenance reports.

---

## 🌟 Key Features

- **Multimodal AI Fusion**:
  - **LSTM Subnet**: Tracks sensor time-series trends over sliding windows of operational cycles.
  - **Graph Neural Network (GNN)**: Captures inter-sensor physical connections and failure propagation topology across engine components (`GCNConv`).
  - **Text Embeddings & Groq LLM**: Converts sensor states into text vectors using `SentenceTransformers` (`all-MiniLM-L6-v2`) and generates automated maintenance diagnostic reports using **Groq LLM**.
- **Interactive Web Dashboard**:
  - User registration & login authentication system (Flask + SQLite).
  - Aircraft engine fleet selector & real-time RUL prediction.
  - Dynamic visual charts: Predicted vs. Actual RUL, Sensor degradation trends, and Model performance comparisons.
  - Automated risk classification (**High Risk**, **Moderate Risk**, **Low Risk**).

---

## 💡 Simple Explanation of Key AI Terms & Concepts

| AI Term / Concept | What it is in Simple Words | What it Does in this Project | Why We Use It |
| :--- | :--- | :--- | :--- |
| **RUL (Remaining Useful Life)** | The countdown clock of a machine. | Calculates how many more flight cycles an aircraft engine can safely operate before it breaks down. | Prevents sudden engine failures mid-flight and avoids costly emergency repairs. |
| **LSTM (Long Short-Term Memory)** | A smart neural network with a long-term memory for time sequences. | Reads sensor data (temperature, pressure, vibration) over a 30-cycle sliding window to detect degradation trends. | Standard neural networks forget history; LSTM remembers patterns over time. |
| **GNN (Graph Neural Network)** | A neural network designed to understand connected networks (like a spider web). | Maps how all 21 engine sensors connect and influence each other when one component starts overheating or wearing down. | Components in a jet engine don't work in isolation; GNN models how failure in part A affects part B. |
| **LLM (Large Language Model - Groq)** | An AI assistant capable of understanding and writing professional text. | Translates complex mathematical sensor models and RUL outputs into plain-English maintenance advice for engineers. | Maintenance engineers need clear action plans (e.g. "Replace bearing in 2 cycles"), not raw matrix numbers. |
| **Sentence Transformers (`all-MiniLM-L6-v2`)** | A text translator that converts words into mathematical vectors. | Converts plain-text sensor status descriptions (e.g. "sensor 11 high, sensor 4 normal") into numerical embedding vectors for AI processing. | Allows our neural network to combine text insights seamlessly with numerical sensor data. |
| **NASA C-MAPSS Dataset** | Official NASA Jet Engine Run-to-Failure dataset. | Contains 21 real sensor channels recorded from engines run until complete mechanical failure. | Provides realistic turbofan engine degradation data for training and benchmark testing. |

---

## 📊 Decoding the "Predict Section" (In Simple Terms)

When you choose an engine and click **"Predict RUL"** on the dashboard, three key numbers appear:

### 1. ⏱️ Predicted RUL (Remaining Useful Life)
* **What it means**: The AI model’s estimation of how many remaining flight cycles this specific engine can safely operate.
* **Example**: If **Predicted RUL = 16 cycles**, the model estimates the engine has 16 operational flights left before requiring mandatory servicing.
* **How it's calculated**: Derived from our **Hybrid GNN + LSTM Neural Network** after analyzing 30 consecutive cycles of 21 sensor streams.

### 2. 🎯 Actual RUL (Ground Truth)
* **What it means**: The real, true number of remaining cycles recorded by NASA test engineers when they ran the engine until it failed.
* **Why it matters**: Acts as the benchmark to measure how close our AI prediction is to reality.

### 3. 📈 Accuracy / Confidence Score (%)
* **What it means**: A percentage indicating how accurate and confident the AI prediction is compared to the real failure point.
* **Formula**:
  $$\text{Accuracy (\%)} = \max\left(0, \; 100 - \frac{|\text{Predicted RUL} - \text{Actual RUL}|}{\text{Actual RUL}} \times 100\right)$$
* **Example**: If Actual RUL = 15 cycles and Predicted RUL = 16 cycles:
  $$\text{Accuracy} = 100 - \frac{|16 - 15|}{15} \times 100 = \mathbf{93.33\%}$$

---

## ⚙️ How the AI Maintenance Explanation is Generated

```
┌────────────────────────┐
│ NASA C-MAPSS Telemetry │
│ (21 Sensor Channels)   │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐     RUL = 16 Cycles
│  Stage 1: AI Model     ├──────────────────────────┐
│  (LSTM + GNN Fusion)   │                          │
└────────────────────────┘                          ▼
                                          ┌────────────────────┐
                                          │  Stage 2: Groq LLM │ ──► Generated Explanation
                                          │  Thermodynamics    │     (+12°C/cycle, 25% RMS, 7% Drop)
                                          └────────────────────┘
```

The AI Maintenance Explanation panel is powered by a **2-Stage Hybrid Process**:

1. **Stage 1: Quantitative RUL & Risk Tier Calculation**
   - The **LSTM + GNN Neural Network** calculates the precise Remaining Useful Life (RUL).
   - A deterministic rule engine evaluates the risk level:
     - **High Risk**: $\text{RUL} \le 25 \text{ cycles}$ (Immediate intervention required).
     - **Moderate Risk**: $25 < \text{RUL} \le 60 \text{ cycles}$ (Scheduled maintenance window).
     - **Low Risk**: $\text{RUL} > 60 \text{ cycles}$ (Normal operation).

2. **Stage 2: Physics-Informed Generative Explanation (Groq LLM)**
   - The exact calculated RUL (e.g. 16) and Risk Level (High Risk) are sent to **Groq LLM**.
   - Groq combines the calculated RUL with turbofan engineering thermodynamics physics to synthesize a detailed technical report:
     - **Temperature Rise (+12 °C/cycle)**: Corresponds to High-Pressure Compressor (`T50` sensor) thermal stress.
     - **Vibration Spike (+25% RMS)**: Reflects turbine shaft bearing wear (`s11` vibration channel).
     - **Pressure Drop (-7%)**: Reflects High-Pressure Turbine tip clearance loss (`P30` discharge pressure).
     - **Action Plan**: Specifies exact replacement cycles for bearings and hot-section components.

---

## 📈 Model Performance & Benchmark Accuracy

Evaluated across 100 test jet engines in the official **NASA C-MAPSS Test Dataset (`RUL_FD001.txt`)**:

| Model Architecture | Mean Absolute Error (MAE) | Accuracy Ranking |
| :--- | :--- | :--- |
| Standard LSTM | ~19.0 cycles error | Baseline |
| Random Forest | ~15.0 cycles error | Intermediate |
| 1D-CNN Model | ~12.0 cycles error | Strong |
| **Our Hybrid GNN + LSTM Model** | **~10.3 cycles error** | 🏆 **Best Performance (Lowest Error)** |

> 💡 **Key Takeaway**: Our Hybrid GNN + LSTM model achieves an error of only **~10.3 cycles**, outperforming standard LSTMs by nearly **45%**.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────┐
                               │  Sensor Time-Series  │
                               └──────────┬───────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             ┌────────────────────┐               ┌──────────────────┐
             │   LSTM Temporal    │               │  GNN Spatial     │
             │   Feature Extractor│               │  Sensor Topology │
             └──────────┬─────────┘               └─────────┬────────┘
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
                                ┌──────────────────┐
                                │ Multimodal Fusion│
                                └─────────┬────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │  RUL Prediction  │
                                └──────────────────┘
                                          │
                                          ▼
                                ┌──────────────────┐
                                │ Groq LLM          │
                                │ Technical Report │
                                └──────────────────┘
```

---

## 🛠️ Installation & Quick Start

### Prerequisites
- Python 3.10+
- PyTorch & PyTorch Geometric
- Flask

### Step 1: Clone Repository
```bash
git clone https://github.com/aditya-johnson/multimodal-predictive-maintenance.git
cd multimodal-predictive-maintenance
```

### Step 2: Install Dependencies
```bash
pip install flask torch torch-geometric sentence-transformers groq python-dotenv pandas numpy scikit-learn matplotlib
```

### Step 3: Set Up Environment Variable
Create a `.env` file in the project root folder:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### Step 4: Run Application
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5050/`**.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
