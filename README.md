# Multimodal Industrial Predictive Maintenance using GNN + LLM Fusion

An end-to-end AI-powered predictive maintenance platform for jet engines (NASA C-MAPSS dataset) that combines **Time-Series Deep Learning (LSTM)**, **Graph Neural Networks (GNN)**, and **Large Language Models (Groq Llama 3.1)** to predict Remaining Useful Life (RUL) and generate automated engineering diagnostic reports.

---

## 🌟 Key Features

- **Multimodal AI Architecture**:
  - **LSTM Subnet**: Captures temporal sensor trend dynamics across sliding windows.
  - **Graph Neural Network (GNN)**: Models inter-sensor spatial relationships and failure propagation topology using dynamic correlation graphs (`GCNConv`).
  - **Text Embeddings & LLM Fusion**: Encodes categorical sensor health states via `SentenceTransformers` (`all-MiniLM-L6-v2`) and generates plain-English maintenance reports powered by **Groq Llama 3.1**.
- **Interactive Web Dashboard**:
  - Flask web application with user authentication (registration, login, session management).
  - Aircraft engine fleet selection and real-time RUL prediction.
  - Dynamic visual charts: Predicted vs. Actual RUL, Sensor degradation trends, and Model performance comparisons.
  - Automated risk classification (**High**, **Moderate**, **Low Risk**).

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
                                │ Groq Llama 3.1   │
                                │ Technical Report │
                                └──────────────────┘
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- PyTorch & PyTorch Geometric
- Flask

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/aditya-johnson/multimodal-predictive-maintenance-gnn-llm.git
   cd multimodal-predictive-maintenance-gnn-llm
   ```

2. **Install dependencies**:
   ```bash
   pip install flask torch torch-geometric sentence-transformers groq pandas numpy scikit-learn matplotlib
   ```

3. **Run the Flask Application**:
   ```bash
   python app.py
   ```

4. **Access Dashboard**:
   Open your browser and navigate to `http://127.0.0.1:5050/` (or `http://127.0.0.1:5000/`).

---

## 📊 Dataset

Built and trained on the **NASA C-MAPSS Jet Engine Run-to-Failure Dataset**:
- **Inputs**: 21 sensor channels (temperatures, pressures, fan speeds, pressure ratios) across operating cycles.
- **Target**: Remaining Useful Life (RUL) in operational cycles.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
