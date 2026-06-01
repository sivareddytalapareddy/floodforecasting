# Flood Forecasting — Federated Learning Architecture

```mermaid
flowchart TD
    %% ─── STYLE DEFINITIONS ────────────────────────────────────────────────────
    classDef server    fill:#0d2137,stroke:#00d4ff,stroke-width:3px,color:#e6edf3,font-weight:bold
    classDef client    fill:#0d1f0d,stroke:#2ecc71,stroke-width:2px,color:#e6edf3
    classDef model     fill:#1a1a2e,stroke:#9b59b6,stroke-width:2px,color:#e6edf3
    classDef data      fill:#1c1005,stroke:#ffa502,stroke-width:2px,color:#e6edf3
    classDef comm      fill:#1a0d0d,stroke:#ff4757,stroke-width:2px,color:#e6edf3
    classDef agg       fill:#0d1a2e,stroke:#00d4ff,stroke-width:3px,color:#00d4ff,font-weight:bold
    classDef alert     fill:#2e0d0d,stroke:#ff4757,stroke-width:2px,color:#ff4757,font-weight:bold
    classDef result    fill:#0d2e0d,stroke:#2ecc71,stroke-width:2px,color:#2ecc71
    classDef preproc   fill:#1a0d2e,stroke:#9b59b6,stroke-width:1px,color:#e6edf3

    %% ─── DATA SOURCES ─────────────────────────────────────────────────────────
    DS1["📄 Station_1.csv\nAlmatti — Upper Basin\n126 yrs · YEAR+JAN–DEC+water_level"]
    DS2["📄 Station_2.csv\nSrisailam — Mid Basin\n126 yrs · YEAR+JAN–DEC+water_level"]
    DS3["📄 Station_3.csv\nPrakasam — Lower Basin\n126 yrs · YEAR+JAN–DEC+water_level"]

    %% ─── PREPROCESSING ────────────────────────────────────────────────────────
    PP1["⚙️ Preprocessing\nMinMaxScaler · shuffle=False\n80% Train / 20% Test"]
    PP2["⚙️ Preprocessing\nMinMaxScaler · shuffle=False\n80% Train / 20% Test"]
    PP3["⚙️ Preprocessing\nMinMaxScaler · shuffle=False\n80% Train / 20% Test"]

    %% ─── CLIENT APPS ──────────────────────────────────────────────────────────
    C1["🖥️ CLIENT 1  ·  Main.py\nTkinter GUI · 4-Phase Workflow\nAlmatti Station Operator"]
    C2["🖥️ CLIENT 2  ·  Main.py\nTkinter GUI · 4-Phase Workflow\nSrisailam Station Operator"]
    C3["🖥️ CLIENT 3  ·  Main.py\nTkinter GUI · 4-Phase Workflow\nPrakasam Station Operator"]

    %% ─── LOCAL MODELS ─────────────────────────────────────────────────────────
    M1A["🧠 FFNN Model\nDense 32→16→1\nAdam+MSE · 100 Epochs\nff_weights.keras"]
    M1B["🧠 CNN-2D Model\nConv2D→MaxPool→Dense\nAdam+MSE · 150 Epochs\nextension_weights.keras"]

    M2A["🧠 FFNN Model\nDense 32→16→1\nAdam+MSE · 100 Epochs\nff_weights.keras"]
    M2B["🧠 CNN-2D Model\nConv2D→MaxPool→Dense\nAdam+MSE · 150 Epochs\nextension_weights.keras"]

    M3A["🧠 FFNN Model\nDense 32→16→1\nAdam+MSE · 100 Epochs\nff_weights.keras"]
    M3B["🧠 CNN-2D Model\nConv2D→MaxPool→Dense\nAdam+MSE · 150 Epochs\nextension_weights.keras"]

    %% ─── COMMUNICATION LAYER ──────────────────────────────────────────────────
    COM["📡 COMMUNICATION LAYER\nTCP Socket · localhost:2222\nJSON + Base64 Encoded Weights\nEOF Delimiter · 64 KB Chunks\n🔒 Weights Only — No Raw Data"]

    %% ─── CENTRAL SERVER ───────────────────────────────────────────────────────
    SRV["🌐 CENTRAL FL SERVER  ·  server1.py\nCustomTkinter Dark UI · 7 Tabs\nDashboard · Station 1/2/3\nAggregation · History · Network Map\nPORT 2222 · psutil Monitor"]

    %% ─── FEDAVG AGGREGATION ───────────────────────────────────────────────────
    FEDAVG["⚙️ FedAvg Algorithm\nnew_weight = mean w1, w2, w3\nLayer-by-layer NumPy average\naggregated_model.keras"]

    %% ─── SERVER EVALUATION ────────────────────────────────────────────────────
    EVAL["📊 Global Model Evaluation\nMSE · RMSE · R² · Accuracy\nMAPE-based Accuracy Score\nInverse MinMaxScaler Transform"]

    %% ─── RESULTS ──────────────────────────────────────────────────────────────
    RES["✅ RESULTS\nCNN-2D: 99.9% Accuracy\nFFNN: Baseline Comparison\nR² · MSE · RMSE per Station"]

    ALERT["🚨 FLOOD ALERT\nWater Level > 25 units\nStation Propagation Timeline\nAlmatti → Srisailam → Prakasam"]

    PDF["📄 PDF Report\nReportLab Auto-Export\nStation Metrics + Charts\nReport_YYYYMMDD_HHMMSS.pdf"]

    %% ─── FLOW ─────────────────────────────────────────────────────────────────
    DS1 --> PP1 --> C1
    DS2 --> PP2 --> C2
    DS3 --> PP3 --> C3

    C1 --> M1A & M1B
    C2 --> M2A & M2B
    C3 --> M3A & M3B

    M1A & M1B -->|"Base64 JSON + EOF"| COM
    M2A & M2B -->|"Base64 JSON + EOF"| COM
    M3A & M3B -->|"Base64 JSON + EOF"| COM

    COM -->|"Decode & Save .keras"| SRV

    SRV -->|"Trigger FedAvg"| FEDAVG
    FEDAVG -->|"Evaluate Global Model"| EVAL
    EVAL --> RES
    EVAL --> ALERT
    EVAL --> PDF

    SRV -->|"Global Model Redistributed"| COM
    COM -->|"Updated Weights"| C1 & C2 & C3

    %% ─── CLASS ASSIGNMENTS ────────────────────────────────────────────────────
    class DS1,DS2,DS3 data
    class PP1,PP2,PP3 preproc
    class C1,C2,C3 client
    class M1A,M1B,M2A,M2B,M3A,M3B model
    class COM comm
    class SRV server
    class FEDAVG agg
    class EVAL result
    class RES result
    class ALERT alert
    class PDF data
```

---

## System Layer Breakdown

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Data** | 6 CSVs · 15 cols · 126 yrs/station | Pandas · NumPy |
| **Preprocessing** | MinMaxScaler · shuffle=False · 80/20 split | Scikit-learn |
| **Client** | Main.py · Tkinter 4-phase GUI · Background threading | Python · Tkinter |
| **Local Models** | FFNN (Dense 32→16→1) · CNN-2D (Conv2D+MaxPool) | Keras / TensorFlow |
| **Communication** | TCP Socket · JSON + Base64 + `<EOF>` · Port 2222 | Python socket |
| **Server** | server1.py · CustomTkinter · 7-tab Dashboard | CustomTkinter |
| **Aggregation** | FedAvg · layer-wise NumPy mean of weights | NumPy |
| **Evaluation** | MSE · RMSE · R² · MAPE-based Accuracy | Scikit-learn |
| **Output** | PDF Report · Flood Alert · Live Dashboard | ReportLab · Matplotlib |

> **Privacy Guarantee:** Raw sensor data **never leaves** the client station. Only encoded model weights travel over the network.
