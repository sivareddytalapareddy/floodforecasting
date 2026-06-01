import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import json
import math
import io

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="FFM: Flood Forecasting Model",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS for Premium Dark Theme ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(52, 152, 219, 0.1) 0%, transparent 60%);
        animation: pulse-bg 4s ease-in-out infinite;
    }
    @keyframes pulse-bg {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        position: relative;
        margin-bottom: 10px;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        font-weight: 400;
        position: relative;
    }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 6px 18px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 15px;
        position: relative;
        letter-spacing: 1px;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(52, 152, 219, 0.15);
        border-color: rgba(52, 152, 219, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #718096;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Phase Headers */
    .phase-header {
        background: linear-gradient(90deg, rgba(52, 152, 219, 0.15), transparent);
        border-left: 4px solid #3498db;
        padding: 12px 20px;
        border-radius: 0 12px 12px 0;
        margin: 25px 0 15px 0;
        font-weight: 700;
        font-size: 1.1rem;
        color: #e2e8f0;
        letter-spacing: 0.5px;
    }

    /* Status Badges */
    .status-normal {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 4px 14px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .status-critical {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 4px 14px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.8rem;
        animation: blink 1s ease-in-out infinite alternate;
    }
    @keyframes blink {
        from { opacity: 1; }
        to { opacity: 0.6; }
    }

    /* Alert Box */
    .flood-alert {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.1));
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
    }
    .safe-alert {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.1));
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
    }

    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }

    /* Progress Steps */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 0;
    }
    .step-dot {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .step-active {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        box-shadow: 0 0 15px rgba(52, 152, 219, 0.4);
    }
    .step-done {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
    }
    .step-pending {
        background: rgba(255, 255, 255, 0.1);
        color: #718096;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }

    /* Data table style */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
def init_state():
    defaults = {
        'dataset': None,
        'filename': '',
        'X': None, 'Y': None,
        'X_train': None, 'y_train': None,
        'X_test': None, 'y_test': None,
        'norm1': None, 'norm2': None,
        'accuracy': [], 'mse_vals': [], 'rmse_vals': [],
        'ffnn_trained': False, 'cnn_trained': False,
        'ffnn_predict': None, 'cnn_predict': None,
        'ffnn_true': None, 'cnn_true': None,
        'current_step': 0,
        'logs': [],
        'ff_history': None,
        'cnn_history': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


def add_log(msg):
    st.session_state.logs.append(msg)


# ─── Hero Section ────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🌊 Flood Forecasting Model</div>
    <div class="hero-subtitle">Federated Learning Powered • CNN + FFNN Deep Neural Networks</div>
    <div class="hero-badge">⚡ REAL-TIME PREDICTION ENGINE</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    st.markdown("---")

    # Progress Tracker
    steps = ["Upload Data", "Preprocess", "Train/Test Split", "Train FFNN", "Train CNN", "Predict"]
    current = st.session_state.current_step
    
    for i, step in enumerate(steps):
        if i < current:
            icon = "✅"
            cls = "step-done"
        elif i == current:
            icon = "🔵"
            cls = "step-active"
        else:
            icon = "⬜"
            cls = "step-pending"
        st.markdown(f"{icon} **Step {i+1}:** {step}")
    
    st.markdown("---")
    
    # Flood Threshold Control
    st.markdown("### ⚙️ Settings")
    flood_threshold = st.slider("🚨 Flood Threshold", 5.0, 100.0, 25.0, 0.5,
                                 help="Water level above this value triggers a flood alert")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if st.session_state.dataset is not None:
        st.metric("Records", len(st.session_state.dataset))
        st.metric("Features", st.session_state.dataset.shape[1] - 2)
    else:
        st.caption("Upload data to see stats")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#718096; font-size:0.75rem;'>"
        "Built with ❤️ using Federated Learning<br>"
        "© 2026 FFM Project"
        "</div>",
        unsafe_allow_html=True
    )


# ─── PHASE 1: DATA PIPELINE ─────────────────────────────────────────────────
st.markdown('<div class="phase-header">📁 PHASE 1: DATA PIPELINE</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1️⃣ Upload Dataset")
    
    # Option to use built-in datasets
    data_source = st.radio("Data Source", ["📂 Built-in Datasets", "📤 Upload CSV"], horizontal=True, label_visibility="collapsed")
    
    if data_source == "📂 Built-in Datasets":
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
        csv_files = []
        if os.path.exists(dataset_dir):
            csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]
        
        if csv_files:
            selected_file = st.selectbox("Select Dataset", csv_files, index=0)
            if st.button("📥 Load Dataset", use_container_width=True, type="primary"):
                filepath = os.path.join(dataset_dir, selected_file)
                st.session_state.dataset = pd.read_csv(filepath)
                st.session_state.dataset.fillna(0, inplace=True)
                st.session_state.filename = selected_file
                st.session_state.current_step = max(st.session_state.current_step, 1)
                add_log(f"✅ Loaded: {selected_file} ({len(st.session_state.dataset)} records)")
                st.rerun()
        else:
            st.warning("No CSV files found in Dataset/ folder")
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        if uploaded_file:
            st.session_state.dataset = pd.read_csv(uploaded_file)
            st.session_state.dataset.fillna(0, inplace=True)
            st.session_state.filename = uploaded_file.name
            st.session_state.current_step = max(st.session_state.current_step, 1)
            add_log(f"✅ Loaded: {uploaded_file.name} ({len(st.session_state.dataset)} records)")

with col2:
    st.markdown("#### 2️⃣ Preprocess Data")
    if st.button("⚙️ Normalize Dataset", use_container_width=True, 
                  disabled=st.session_state.dataset is None, type="primary"):
        from sklearn.preprocessing import MinMaxScaler
        
        norm1 = MinMaxScaler(feature_range=(0, 1))
        norm2 = MinMaxScaler(feature_range=(0, 1))
        
        dataset_vals = st.session_state.dataset.values
        X = dataset_vals[:, 2:dataset_vals.shape[1]-1]
        Y = dataset_vals[:, dataset_vals.shape[1]-1]
        Y = Y.reshape(-1, 1)
        
        X = norm1.fit_transform(X)
        Y = norm2.fit_transform(Y)
        
        st.session_state.X = X
        st.session_state.Y = Y
        st.session_state.norm1 = norm1
        st.session_state.norm2 = norm2
        st.session_state.current_step = max(st.session_state.current_step, 2)
        add_log("✅ Preprocessing complete: MinMaxScaler normalization applied")
        st.rerun()
    
    if st.session_state.X is not None:
        st.success(f"Normalized: {st.session_state.X.shape[0]} samples × {st.session_state.X.shape[1]} features")

with col3:
    st.markdown("#### 3️⃣ Train/Test Split")
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    if st.button("✂️ Split Dataset", use_container_width=True, 
                  disabled=st.session_state.X is None, type="primary"):
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.X, st.session_state.Y, 
            test_size=test_size, shuffle=False
        )
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.current_step = max(st.session_state.current_step, 3)
        add_log(f"✅ Split: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
        st.rerun()
    
    if st.session_state.X_train is not None:
        st.success(f"Train: {st.session_state.X_train.shape[0]} | Test: {st.session_state.X_test.shape[0]}")

# Show Dataset Preview
if st.session_state.dataset is not None:
    with st.expander("📋 Dataset Preview", expanded=False):
        st.dataframe(st.session_state.dataset.head(20), use_container_width=True)


# ─── PHASE 2: MODEL TRAINING ────────────────────────────────────────────────
st.markdown('<div class="phase-header">🧠 PHASE 2: MODEL TRAINING</div>', unsafe_allow_html=True)

train_col1, train_col2 = st.columns(2)

with train_col1:
    st.markdown("#### 🔮 FFNN (Feed-Forward Neural Network)")
    
    ffnn_epochs = st.number_input("Epochs", 10, 500, 100, key="ffnn_ep")
    
    if st.button("🚀 Train FFNN", use_container_width=True, 
                  disabled=st.session_state.X_train is None, type="primary"):
        from sklearn.metrics import mean_squared_error, r2_score
        
        with st.spinner("Training FFNN... This may take a moment."):
            try:
                # Check for cached model
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'ff_weights.keras')
                history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'ff_history.json')
                
                if os.path.exists(model_path):
                    from keras.models import load_model
                    model = load_model(model_path)
                    add_log("📦 FFNN: Loaded from cache (model/ff_weights.keras)")
                    
                    # Load history if available
                    if os.path.exists(history_path):
                        with open(history_path, 'r') as f:
                            st.session_state.ff_history = json.load(f)
                else:
                    from keras.models import Sequential
                    from keras.layers import Dense
                    from keras.callbacks import ModelCheckpoint
                    
                    model = Sequential()
                    model.add(Dense(32, input_shape=(st.session_state.X.shape[1],)))
                    model.add(Dense(16, activation="relu"))
                    model.add(Dense(units=1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    os.makedirs('model', exist_ok=True)
                    model_check = ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True)
                    
                    history = model.fit(
                        st.session_state.X_train, st.session_state.y_train,
                        epochs=int(ffnn_epochs), batch_size=8,
                        validation_data=(st.session_state.X_test, st.session_state.y_test),
                        callbacks=[model_check], verbose=0
                    )
                    
                    st.session_state.ff_history = {
                        'loss': [float(v) for v in history.history['loss']],
                        'val_loss': [float(v) for v in history.history['val_loss']]
                    }
                    with open(history_path, 'w') as f:
                        json.dump(st.session_state.ff_history, f)
                    
                    model = load_model(model_path) if os.path.exists(model_path) else model
                    add_log(f"✅ FFNN: Trained for {int(ffnn_epochs)} epochs")
                
                # Predict
                predict = model.predict(st.session_state.X_test)
                predict_inv = st.session_state.norm2.inverse_transform(np.abs(predict))
                test_inv = st.session_state.norm2.inverse_transform(st.session_state.y_test)
                
                r2 = r2_score(test_inv, predict_inv)
                mse_val = mean_squared_error(test_inv, predict_inv)
                rmse_val = math.sqrt(mse_val)
                acc = max(0, 100 - (rmse_val / np.mean(test_inv) * 100))
                
                st.session_state.ffnn_trained = True
                st.session_state.ffnn_predict = predict_inv
                st.session_state.ffnn_true = test_inv
                st.session_state.accuracy = [acc]
                st.session_state.mse_vals = [mse_val]
                st.session_state.rmse_vals = [rmse_val]
                st.session_state.current_step = max(st.session_state.current_step, 4)
                add_log(f"📊 FFNN Results: Acc={acc:.2f}% | MSE={mse_val:.4f} | R²={r2:.4f}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Training error: {e}")
                add_log(f"❌ FFNN Error: {e}")

with train_col2:
    st.markdown("#### 🧬 CNN2D (Convolutional Neural Network)")
    
    cnn_epochs = st.number_input("Epochs", 10, 500, 150, key="cnn_ep")
    
    if st.button("🚀 Train CNN2D", use_container_width=True, 
                  disabled=st.session_state.X_train is None, type="primary"):
        from sklearn.metrics import mean_squared_error, r2_score
        
        with st.spinner("Training CNN2D... This may take a moment."):
            try:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'extension_weights.keras')
                history_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'cnn_history.json')
                
                if os.path.exists(model_path):
                    from keras.models import load_model
                    model = load_model(model_path)
                    add_log("📦 CNN2D: Loaded from cache (model/extension_weights.keras)")
                    
                    if os.path.exists(history_path):
                        with open(history_path, 'r') as f:
                            st.session_state.cnn_history = json.load(f)
                else:
                    from keras.models import Sequential
                    from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D
                    from keras.callbacks import ModelCheckpoint
                    
                    X_train1 = st.session_state.X_train.reshape(
                        st.session_state.X_train.shape[0], st.session_state.X_train.shape[1], 1, 1)
                    X_test1 = st.session_state.X_test.reshape(
                        st.session_state.X_test.shape[0], st.session_state.X_test.shape[1], 1, 1)
                    
                    model = Sequential()
                    model.add(Convolution2D(32, (3, 1), input_shape=(12, 1, 1), activation='relu', padding='same'))
                    model.add(MaxPooling2D(pool_size=(2, 1)))
                    model.add(Convolution2D(64, (3, 1), activation='relu', padding='same'))
                    model.add(MaxPooling2D(pool_size=(2, 1)))
                    model.add(Flatten())
                    model.add(Dense(100, activation='relu'))
                    model.add(Dense(1, activation='linear'))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    os.makedirs('model', exist_ok=True)
                    model_check = ModelCheckpoint(filepath=model_path, verbose=0, save_best_only=True)
                    
                    history = model.fit(
                        X_train1, st.session_state.y_train,
                        epochs=int(cnn_epochs), batch_size=8,
                        validation_data=(X_test1, st.session_state.y_test),
                        callbacks=[model_check], verbose=0
                    )
                    
                    st.session_state.cnn_history = {
                        'loss': [float(v) for v in history.history['loss']],
                        'val_loss': [float(v) for v in history.history['val_loss']]
                    }
                    with open(history_path, 'w') as f:
                        json.dump(st.session_state.cnn_history, f)
                    
                    model = load_model(model_path) if os.path.exists(model_path) else model
                    add_log(f"✅ CNN2D: Trained for {int(cnn_epochs)} epochs")
                
                # Predict
                X_test_cnn = st.session_state.X_test.reshape(
                    st.session_state.X_test.shape[0], st.session_state.X_test.shape[1], 1, 1)
                predict = model.predict(X_test_cnn)
                predict_inv = st.session_state.norm2.inverse_transform(np.abs(predict))
                test_inv = st.session_state.norm2.inverse_transform(st.session_state.y_test)
                
                r2 = r2_score(test_inv, predict_inv)
                mse_val = mean_squared_error(test_inv, predict_inv)
                rmse_val = math.sqrt(mse_val)
                acc = max(0, 100 - (rmse_val / np.mean(test_inv) * 100))
                
                st.session_state.cnn_trained = True
                st.session_state.cnn_predict = predict_inv
                st.session_state.cnn_true = test_inv
                
                if len(st.session_state.accuracy) < 2:
                    st.session_state.accuracy.append(acc)
                    st.session_state.mse_vals.append(mse_val)
                    st.session_state.rmse_vals.append(rmse_val)
                else:
                    st.session_state.accuracy[1] = acc
                    st.session_state.mse_vals[1] = mse_val
                    st.session_state.rmse_vals[1] = rmse_val
                
                st.session_state.current_step = max(st.session_state.current_step, 5)
                add_log(f"📊 CNN2D Results: Acc={acc:.2f}% | MSE={mse_val:.4f} | R²={r2:.4f}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Training error: {e}")
                add_log(f"❌ CNN2D Error: {e}")

# ─── Training Results Display ────────────────────────────────────────────────
if st.session_state.ffnn_trained or st.session_state.cnn_trained:
    st.markdown('<div class="phase-header">📊 TRAINING RESULTS</div>', unsafe_allow_html=True)
    
    # Metric Cards
    if st.session_state.ffnn_trained and st.session_state.cnn_trained:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.accuracy[0]:.1f}%</div>
                <div class="metric-label">FFNN Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.accuracy[1]:.1f}%</div>
                <div class="metric-label">CNN2D Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.mse_vals[0]:.4f}</div>
                <div class="metric-label">FFNN MSE</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{st.session_state.mse_vals[1]:.4f}</div>
                <div class="metric-label">CNN2D MSE</div>
            </div>""", unsafe_allow_html=True)
    
    # Prediction Plots
    plot_col1, plot_col2 = st.columns(2)
    
    if st.session_state.ffnn_trained:
        with plot_col1:
            fig_ffnn = go.Figure()
            fig_ffnn.add_trace(go.Scatter(
                y=st.session_state.ffnn_true.flatten(), mode='lines',
                name='True Water Level', line=dict(color='#e74c3c', width=2)
            ))
            fig_ffnn.add_trace(go.Scatter(
                y=st.session_state.ffnn_predict.flatten(), mode='lines',
                name='FFNN Predicted', line=dict(color='#2ecc71', width=2)
            ))
            fig_ffnn.update_layout(
                title="FFNN: Prediction vs True",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=400,
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_ffnn, use_container_width=True)
    
    if st.session_state.cnn_trained:
        with plot_col2:
            fig_cnn = go.Figure()
            fig_cnn.add_trace(go.Scatter(
                y=st.session_state.cnn_true.flatten(), mode='lines',
                name='True Water Level', line=dict(color='#e74c3c', width=2)
            ))
            fig_cnn.add_trace(go.Scatter(
                y=st.session_state.cnn_predict.flatten(), mode='lines',
                name='CNN2D Predicted', line=dict(color='#3498db', width=2)
            ))
            fig_cnn.update_layout(
                title="CNN2D: Prediction vs True",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=400,
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_cnn, use_container_width=True)
    
    # Training History Plots
    if st.session_state.ff_history or st.session_state.cnn_history:
        with st.expander("📈 Training Loss History", expanded=False):
            hist_col1, hist_col2 = st.columns(2)
            
            if st.session_state.ff_history:
                with hist_col1:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(y=st.session_state.ff_history['loss'], 
                                               name='Train Loss', line=dict(color='cyan')))
                    fig_h.add_trace(go.Scatter(y=st.session_state.ff_history['val_loss'], 
                                               name='Val Loss', line=dict(color='orange')))
                    fig_h.update_layout(title="FFNN Loss Curve", template="plotly_dark",
                                        paper_bgcolor='rgba(0,0,0,0)', height=300)
                    st.plotly_chart(fig_h, use_container_width=True)
            
            if st.session_state.cnn_history:
                with hist_col2:
                    fig_h2 = go.Figure()
                    fig_h2.add_trace(go.Scatter(y=st.session_state.cnn_history['loss'], 
                                                name='Train Loss', line=dict(color='cyan')))
                    fig_h2.add_trace(go.Scatter(y=st.session_state.cnn_history['val_loss'], 
                                                name='Val Loss', line=dict(color='orange')))
                    fig_h2.update_layout(title="CNN2D Loss Curve", template="plotly_dark",
                                          paper_bgcolor='rgba(0,0,0,0)', height=300)
                    st.plotly_chart(fig_h2, use_container_width=True)
    
    # Comparison Bar Chart
    if st.session_state.ffnn_trained and st.session_state.cnn_trained:
        with st.expander("📊 Model Comparison", expanded=True):
            fig_comp = make_subplots(rows=1, cols=3, subplot_titles=("Accuracy (%)", "MSE", "RMSE"))
            
            models = ['FFNN', 'CNN2D']
            colors = ['#9b59b6', '#3498db']
            
            fig_comp.add_trace(go.Bar(x=models, y=st.session_state.accuracy, 
                                       marker_color=colors, text=[f"{v:.1f}%" for v in st.session_state.accuracy],
                                       textposition='auto'), row=1, col=1)
            fig_comp.add_trace(go.Bar(x=models, y=st.session_state.mse_vals, 
                                       marker_color=colors, text=[f"{v:.4f}" for v in st.session_state.mse_vals],
                                       textposition='auto'), row=1, col=2)
            fig_comp.add_trace(go.Bar(x=models, y=st.session_state.rmse_vals, 
                                       marker_color=colors, text=[f"{v:.4f}" for v in st.session_state.rmse_vals],
                                       textposition='auto'), row=1, col=3)
            
            fig_comp.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=350, showlegend=False,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ─── PHASE 3: FLOOD FORECAST ────────────────────────────────────────────────
st.markdown('<div class="phase-header">🌊 PHASE 3: FLOOD FORECAST & PREDICTION</div>', unsafe_allow_html=True)

if st.session_state.cnn_trained:
    pred_source = st.radio("Prediction Data", ["📂 Built-in Test Data", "📤 Upload New CSV"], horizontal=True)
    
    run_predict = False
    pred_data = None
    pred_name = ""
    
    if pred_source == "📂 Built-in Test Data":
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')] if os.path.exists(dataset_dir) else []
        selected_pred = st.selectbox("Select Test Data", csv_files, key="pred_file")
        if st.button("🌊 Run Flood Forecast", use_container_width=True, type="primary"):
            pred_data = pd.read_csv(os.path.join(dataset_dir, selected_pred))
            pred_data.fillna(0, inplace=True)
            pred_name = selected_pred
            run_predict = True
    else:
        pred_upload = st.file_uploader("Upload Prediction CSV", type=['csv'], key="pred_upload")
        if pred_upload and st.button("🌊 Run Flood Forecast", use_container_width=True, type="primary"):
            pred_data = pd.read_csv(pred_upload)
            pred_data.fillna(0, inplace=True)
            pred_name = pred_upload.name
            run_predict = True
    
    if run_predict and pred_data is not None:
        with st.spinner("Running flood prediction..."):
            try:
                from keras.models import load_model
                
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'extension_weights.keras')
                model = load_model(model_path)
                
                dataset_values = pred_data.values
                features_start = 2
                features_end = dataset_values.shape[1]
                
                try:
                    expected_features = len(st.session_state.norm1.min_)
                    current_features = features_end - features_start
                    if current_features == expected_features + 1:
                        features_end = features_end - 1
                except:
                    features_end = features_end - 1
                
                X_pred = dataset_values[:, features_start:features_end]
                X_pred = st.session_state.norm1.transform(X_pred)
                X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1, 1))
                
                predictions = model.predict(X_pred)
                predictions = st.session_state.norm2.inverse_transform(predictions)
                
                # Extract years
                try:
                    years = dataset_values[:, 1].flatten().astype(int)
                    time_label = "Year"
                except:
                    years = list(range(1, len(predictions) + 1))
                    time_label = "Time Step"
                
                flood_detected = any(predictions.flatten() > flood_threshold)
                
                # Alert
                if flood_detected:
                    st.markdown(f"""
                    <div class="flood-alert">
                        <h3>🚨 CRITICAL FLOOD ALERT</h3>
                        <p>Predicted water levels exceed the critical threshold of <strong>{flood_threshold}</strong> units!</p>
                        <p>Immediate attention and preventive measures are required.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="safe-alert">
                        <h3>✅ SAFE — Normal Water Levels</h3>
                        <p>All predicted water levels are within the safe range (below {flood_threshold} units).</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Forecast Chart
                fig_forecast = go.Figure()
                
                # Safe zone
                fig_forecast.add_hrect(
                    y0=min(predictions.min(), 0), y1=flood_threshold,
                    fillcolor="rgba(46, 204, 113, 0.1)", line_width=0,
                    annotation_text="Safe Zone", annotation_position="bottom left"
                )
                # Danger zone
                fig_forecast.add_hrect(
                    y0=flood_threshold, y1=max(predictions.max(), flood_threshold + 10),
                    fillcolor="rgba(231, 76, 60, 0.15)", line_width=0,
                    annotation_text="⚠️ Danger Zone", annotation_position="top left"
                )
                
                # Threshold line
                fig_forecast.add_hline(
                    y=flood_threshold, line_dash="dash",
                    line_color="#c0392b", line_width=2,
                    annotation_text=f"Flood Threshold ({flood_threshold})"
                )
                
                # Prediction line
                fig_forecast.add_trace(go.Scatter(
                    x=years, y=predictions.flatten(),
                    mode='lines+markers',
                    name='Predicted Water Level',
                    line=dict(color='#3498db', width=3),
                    marker=dict(size=8, color='#3498db',
                                line=dict(width=2, color='white')),
                    fill='tozeroy',
                    fillcolor='rgba(52, 152, 219, 0.05)'
                ))
                
                # Peak annotation
                max_val = predictions.max()
                max_idx = np.argmax(predictions)
                if max_val > flood_threshold:
                    fig_forecast.add_annotation(
                        x=years[max_idx], y=float(max_val),
                        text=f"🔺 PEAK FLOOD<br>{max_val:.1f} units",
                        showarrow=True, arrowhead=2,
                        font=dict(size=14, color='#e74c3c'),
                        bgcolor='rgba(231,76,60,0.1)',
                        bordercolor='#e74c3c'
                    )
                
                fig_forecast.update_layout(
                    title=f"🌊 Flood Forecast Analysis: {pred_name}",
                    xaxis_title=time_label,
                    yaxis_title="Water Level (Units)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26,26,46,0.8)',
                    height=500,
                    margin=dict(l=50, r=30, t=60, b=50),
                    font=dict(family="Inter")
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Detailed Results Table
                with st.expander("📋 Detailed Forecast Results", expanded=True):
                    results_df = pd.DataFrame({
                        time_label: years,
                        'Water Level': predictions.flatten().round(2),
                        'Status': ['🔴 CRITICAL' if v > flood_threshold else '🟢 NORMAL' 
                                   for v in predictions.flatten()]
                    })
                    st.dataframe(results_df, use_container_width=True, height=300)
                
                add_log(f"🌊 Forecast complete for {pred_name}: {'FLOOD ALERT' if flood_detected else 'Normal'}")
                
            except Exception as e:
                st.error(f"Prediction error: {e}")
                add_log(f"❌ Prediction Error: {e}")
else:
    st.info("⬆️ Please train at least the CNN2D model to enable flood forecasting.")


# ─── System Logs ─────────────────────────────────────────────────────────────
with st.expander("🖥️ System Logs", expanded=False):
    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-20:]):
            st.text(log)
    else:
        st.caption("No logs yet. Start by uploading a dataset.")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p style='font-size: 0.9rem;'>
        🌊 <strong>FFM: Flood Forecasting Model</strong> — Powered by Federated Learning & Deep Neural Networks
    </p>
    <p style='font-size: 0.75rem;'>
        CNN2D + FFNN Architecture • MinMaxScaler Normalization • Real-time Prediction
    </p>
</div>
""", unsafe_allow_html=True)
