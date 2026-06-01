import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="FFM: Flood Forecasting Model",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px; padding: 40px; margin-bottom: 30px;
        text-align: center; position: relative; overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    .hero-container::before {
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(52, 152, 219, 0.1) 0%, transparent 60%);
        animation: pulse-bg 4s ease-in-out infinite;
    }
    @keyframes pulse-bg {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
        background-size: 200% 200%;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        animation: gradient-shift 3s ease infinite;
        position: relative; margin-bottom: 10px;
    }
    @keyframes gradient-shift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-subtitle { color: #a0aec0; font-size: 1.1rem; font-weight: 400; position: relative; }
    .hero-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 6px 18px; border-radius: 50px;
        font-size: 0.8rem; font-weight: 600; margin-top: 15px;
        position: relative; letter-spacing: 1px;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px; padding: 24px; text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(52, 152, 219, 0.15);
        border-color: rgba(52, 152, 219, 0.3);
    }
    .metric-value {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #718096; font-size: 0.85rem; font-weight: 500;
        margin-top: 8px; text-transform: uppercase; letter-spacing: 1px;
    }

    .phase-header {
        background: linear-gradient(90deg, rgba(52, 152, 219, 0.15), transparent);
        border-left: 4px solid #3498db; padding: 12px 20px;
        border-radius: 0 12px 12px 0; margin: 25px 0 15px 0;
        font-weight: 700; font-size: 1.1rem; color: #e2e8f0; letter-spacing: 0.5px;
    }

    .flood-alert {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.15), rgba(192, 57, 43, 0.1));
        border: 1px solid rgba(231, 76, 60, 0.3);
        border-radius: 16px; padding: 20px; margin: 15px 0;
    }
    .safe-alert {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.15), rgba(39, 174, 96, 0.1));
        border: 1px solid rgba(46, 204, 113, 0.3);
        border-radius: 16px; padding: 20px; margin: 15px 0;
    }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29, #1a1a2e); }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Load Precomputed Data ───────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PRECOMPUTED = os.path.join(BASE, 'precomputed')

@st.cache_data
def load_precomputed():
    data = {}
    if os.path.exists(PRECOMPUTED):
        for f in os.listdir(PRECOMPUTED):
            if f.endswith('_results.json'):
                with open(os.path.join(PRECOMPUTED, f), 'r') as fp:
                    result = json.load(fp)
                    data[result['dataset_name']] = result
    return data

@st.cache_data
def load_history(name):
    path = os.path.join(PRECOMPUTED, name)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    # Fallback to model directory
    path2 = os.path.join(BASE, 'model', name)
    if os.path.exists(path2):
        with open(path2, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset(name):
    path = os.path.join(BASE, 'Dataset', name)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.fillna(0, inplace=True)
        return df
    return None

all_results = load_precomputed()
ff_history = load_history('ff_history.json')
cnn_history = load_history('cnn_history.json')


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

    available_datasets = list(all_results.keys()) if all_results else []
    
    if available_datasets:
        selected_dataset = st.selectbox("📂 Select Dataset", available_datasets, index=0)
        current = all_results.get(selected_dataset, {})
    else:
        selected_dataset = None
        current = {}
        st.warning("No precomputed data found")

    st.markdown("---")
    flood_threshold = st.slider("🚨 Flood Threshold", 5.0, 100.0, 25.0, 0.5,
                                 help="Water level above this triggers a flood alert")

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if current:
        st.metric("Records", current.get('total_records', 0))
        st.metric("Features", current.get('total_features', 0))
        st.metric("Train Size", current.get('train_size', 0))
        st.metric("Test Size", current.get('test_size', 0))

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#718096; font-size:0.75rem;'>"
        "Built with ❤️ using Federated Learning<br>"
        "© 2026 FFM Project"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Dataset Preview ────────────────────────────────────────────────────────
if selected_dataset:
    st.markdown('<div class="phase-header">📁 DATASET OVERVIEW</div>', unsafe_allow_html=True)
    
    df = load_dataset(selected_dataset)
    if df is not None:
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Total Records</div>
            </div>""", unsafe_allow_html=True)
        with col_info2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.shape[1] - 2}</div>
                <div class="metric-label">Input Features</div>
            </div>""", unsafe_allow_html=True)
        with col_info3:
            split_pct = f"{current.get('train_size', 0)}/{current.get('test_size', 0)}"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{split_pct}</div>
                <div class="metric-label">Train / Test Split</div>
            </div>""", unsafe_allow_html=True)
        
        with st.expander("📋 Dataset Preview", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)


# ─── Model Results ───────────────────────────────────────────────────────────
ffnn = current.get('ffnn', None)
cnn = current.get('cnn', None)

if ffnn or cnn:
    st.markdown('<div class="phase-header">🧠 MODEL TRAINING RESULTS</div>', unsafe_allow_html=True)
    
    # Metric Cards
    if ffnn and cnn:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ffnn['accuracy']:.1f}%</div>
                <div class="metric-label">FFNN Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cnn['accuracy']:.1f}%</div>
                <div class="metric-label">CNN2D Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ffnn['r2']:.4f}</div>
                <div class="metric-label">FFNN R² Score</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{cnn['r2']:.4f}</div>
                <div class="metric-label">CNN2D R² Score</div>
            </div>""", unsafe_allow_html=True)
    elif ffnn:
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ffnn['accuracy']:.1f}%</div>
                <div class="metric-label">FFNN Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{ffnn['r2']:.4f}</div>
                <div class="metric-label">FFNN R² Score</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")  # spacing

    # ─── Prediction Charts ───────────────────────────────────────────────
    plot_col1, plot_col2 = st.columns(2)

    if ffnn:
        with plot_col1:
            fig_ffnn = go.Figure()
            fig_ffnn.add_trace(go.Scatter(
                y=ffnn['true_values'], mode='lines',
                name='True Water Level', line=dict(color='#e74c3c', width=2)
            ))
            fig_ffnn.add_trace(go.Scatter(
                y=ffnn['predictions'], mode='lines',
                name='FFNN Predicted', line=dict(color='#2ecc71', width=2)
            ))
            fig_ffnn.update_layout(
                title=dict(text="FFNN: Prediction vs True", font=dict(size=16)),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=400,
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="Test Sample Index",
                yaxis_title="Water Level"
            )
            st.plotly_chart(fig_ffnn, use_container_width=True)

    if cnn:
        with plot_col2:
            fig_cnn = go.Figure()
            fig_cnn.add_trace(go.Scatter(
                y=cnn['true_values'], mode='lines',
                name='True Water Level', line=dict(color='#e74c3c', width=2)
            ))
            fig_cnn.add_trace(go.Scatter(
                y=cnn['predictions'], mode='lines',
                name='CNN2D Predicted', line=dict(color='#3498db', width=2)
            ))
            fig_cnn.update_layout(
                title=dict(text="CNN2D: Prediction vs True", font=dict(size=16)),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=400,
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_title="Test Sample Index",
                yaxis_title="Water Level"
            )
            st.plotly_chart(fig_cnn, use_container_width=True)

    # ─── Training History ────────────────────────────────────────────────
    if ff_history or cnn_history:
        with st.expander("📈 Training Loss History", expanded=False):
            hist_col1, hist_col2 = st.columns(2)
            
            if ff_history:
                with hist_col1:
                    fig_h = go.Figure()
                    fig_h.add_trace(go.Scatter(y=ff_history.get('loss', []),
                                               name='Train Loss', line=dict(color='cyan', width=2)))
                    fig_h.add_trace(go.Scatter(y=ff_history.get('val_loss', []),
                                               name='Val Loss', line=dict(color='orange', width=2)))
                    fig_h.update_layout(title="FFNN Loss Curve", template="plotly_dark",
                                        paper_bgcolor='rgba(0,0,0,0)', height=300,
                                        xaxis_title="Epoch", yaxis_title="Loss (MSE)")
                    st.plotly_chart(fig_h, use_container_width=True)

            if cnn_history:
                with hist_col2:
                    fig_h2 = go.Figure()
                    fig_h2.add_trace(go.Scatter(y=cnn_history.get('loss', []),
                                                name='Train Loss', line=dict(color='cyan', width=2)))
                    fig_h2.add_trace(go.Scatter(y=cnn_history.get('val_loss', []),
                                                name='Val Loss', line=dict(color='orange', width=2)))
                    fig_h2.update_layout(title="CNN2D Loss Curve", template="plotly_dark",
                                          paper_bgcolor='rgba(0,0,0,0)', height=300,
                                          xaxis_title="Epoch", yaxis_title="Loss (MSE)")
                    st.plotly_chart(fig_h2, use_container_width=True)

    # ─── Comparison Chart ────────────────────────────────────────────────
    if ffnn and cnn:
        with st.expander("📊 Model Comparison", expanded=True):
            models = ['FFNN', 'CNN2D']
            colors = ['#9b59b6', '#3498db']
            
            fig_comp = make_subplots(
                rows=1, cols=4,
                subplot_titles=("Accuracy (%)", "MSE", "RMSE", "R² Score")
            )
            
            accs = [ffnn['accuracy'], cnn['accuracy']]
            mses = [ffnn['mse'], cnn['mse']]
            rmses = [ffnn['rmse'], cnn['rmse']]
            r2s = [ffnn['r2'], cnn['r2']]
            
            fig_comp.add_trace(go.Bar(x=models, y=accs, marker_color=colors,
                                       text=[f"{v:.1f}%" for v in accs], textposition='auto',
                                       showlegend=False), row=1, col=1)
            fig_comp.add_trace(go.Bar(x=models, y=mses, marker_color=colors,
                                       text=[f"{v:.2f}" for v in mses], textposition='auto',
                                       showlegend=False), row=1, col=2)
            fig_comp.add_trace(go.Bar(x=models, y=rmses, marker_color=colors,
                                       text=[f"{v:.2f}" for v in rmses], textposition='auto',
                                       showlegend=False), row=1, col=3)
            fig_comp.add_trace(go.Bar(x=models, y=r2s, marker_color=colors,
                                       text=[f"{v:.4f}" for v in r2s], textposition='auto',
                                       showlegend=False), row=1, col=4)
            
            fig_comp.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                height=350, showlegend=False,
                margin=dict(l=40, r=20, t=50, b=40)
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # Detailed metrics table
        with st.expander("📋 Detailed Metrics", expanded=False):
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy (%)', 'MSE', 'RMSE', 'R² Score'],
                'FFNN': [f"{ffnn['accuracy']:.2f}", f"{ffnn['mse']:.6f}", 
                         f"{ffnn['rmse']:.6f}", f"{ffnn['r2']:.6f}"],
                'CNN2D': [f"{cnn['accuracy']:.2f}", f"{cnn['mse']:.6f}", 
                          f"{cnn['rmse']:.6f}", f"{cnn['r2']:.6f}"],
                'Winner': [
                    '🏆 ' + ('FFNN' if ffnn['accuracy'] > cnn['accuracy'] else 'CNN2D'),
                    '🏆 ' + ('FFNN' if ffnn['mse'] < cnn['mse'] else 'CNN2D'),
                    '🏆 ' + ('FFNN' if ffnn['rmse'] < cnn['rmse'] else 'CNN2D'),
                    '🏆 ' + ('FFNN' if ffnn['r2'] > cnn['r2'] else 'CNN2D'),
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)


# ─── FLOOD FORECAST ─────────────────────────────────────────────────────────
st.markdown('<div class="phase-header">🌊 FLOOD FORECAST & PREDICTION</div>', unsafe_allow_html=True)

# Check for forecast data
forecast_data = current.get('forecast', None)
forecast_now = current.get('forecast_now', None)

available_forecasts = {}
if forecast_data:
    available_forecasts[forecast_data.get('source', 'testData.csv')] = forecast_data
if forecast_now:
    available_forecasts[forecast_now.get('source', 'test_now.csv')] = forecast_now

if available_forecasts:
    forecast_choice = st.selectbox("🗂️ Select Forecast Data", list(available_forecasts.keys()))
    
    if st.button("🌊 Run Flood Forecast", use_container_width=True, type="primary"):
        fdata = available_forecasts[forecast_choice]
        predictions = np.array(fdata['predictions'])
        years = fdata['years']
        
        flood_detected = any(predictions > flood_threshold)
        
        # Alert
        if flood_detected:
            flood_count = sum(1 for p in predictions if p > flood_threshold)
            max_level = max(predictions)
            st.markdown(f"""
            <div class="flood-alert">
                <h3>🚨 CRITICAL FLOOD ALERT</h3>
                <p>⚠️ <strong>{flood_count} out of {len(predictions)}</strong> predictions exceed the critical threshold of <strong>{flood_threshold:.1f}</strong> units!</p>
                <p>📈 Peak water level: <strong>{max_level:.1f}</strong> units</p>
                <p>Immediate attention and preventive measures are required.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                <h3>✅ SAFE — Normal Water Levels</h3>
                <p>All predicted water levels are within the safe range (below <strong>{flood_threshold:.1f}</strong> units).</p>
                <p>No immediate action required.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forecast Chart
        fig_forecast = go.Figure()
        
        # Safe/Danger zones
        y_min = min(float(predictions.min()), 0)
        y_max = max(float(predictions.max()), flood_threshold + 10)
        
        fig_forecast.add_hrect(
            y0=y_min, y1=flood_threshold,
            fillcolor="rgba(46, 204, 113, 0.08)", line_width=0,
            annotation_text="✅ Safe Zone", annotation_position="bottom left",
            annotation_font_color="#2ecc71"
        )
        fig_forecast.add_hrect(
            y0=flood_threshold, y1=y_max,
            fillcolor="rgba(231, 76, 60, 0.1)", line_width=0,
            annotation_text="⚠️ Danger Zone", annotation_position="top left",
            annotation_font_color="#e74c3c"
        )
        
        # Threshold line
        fig_forecast.add_hline(
            y=flood_threshold, line_dash="dash",
            line_color="#c0392b", line_width=2,
            annotation_text=f"Flood Threshold ({flood_threshold:.1f})",
            annotation_font_color="#e74c3c"
        )
        
        # Color code markers by status
        marker_colors = ['#e74c3c' if p > flood_threshold else '#2ecc71' for p in predictions]
        
        fig_forecast.add_trace(go.Scatter(
            x=years, y=predictions.tolist(),
            mode='lines+markers',
            name='Predicted Water Level',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10, color=marker_colors,
                        line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(52, 152, 219, 0.05)',
            hovertemplate="<b>Year %{x}</b><br>Water Level: %{y:.2f}<extra></extra>"
        ))
        
        # Peak annotation
        max_val = float(predictions.max())
        max_idx = int(np.argmax(predictions))
        fig_forecast.add_annotation(
            x=years[max_idx], y=max_val,
            text=f"📍 Peak: {max_val:.1f}",
            showarrow=True, arrowhead=2, arrowcolor='#e74c3c',
            font=dict(size=13, color='#e74c3c', family='Inter'),
            bgcolor='rgba(231,76,60,0.1)', bordercolor='#e74c3c',
            borderpad=4
        )
        
        fig_forecast.update_layout(
            title=dict(text=f"🌊 Flood Forecast: {forecast_choice}", font=dict(size=18, family='Inter')),
            xaxis_title="Year",
            yaxis_title="Water Level (Units)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)',
            height=500,
            margin=dict(l=50, r=30, t=60, b=50),
            font=dict(family="Inter"),
            yaxis=dict(range=[y_min - 5, y_max + 5])
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Results Table
        with st.expander("📋 Detailed Forecast Results", expanded=True):
            results_df = pd.DataFrame({
                'Year': years,
                'Water Level': [f"{p:.2f}" for p in predictions],
                'Status': ['🔴 CRITICAL' if p > flood_threshold else '🟢 NORMAL' for p in predictions],
                'Deviation': [f"+{p - flood_threshold:.2f}" if p > flood_threshold else f"{p - flood_threshold:.2f}" for p in predictions]
            })
            st.dataframe(results_df, use_container_width=True, hide_index=True)
else:
    st.info("⚠️ No precomputed forecast data available for this dataset. Run `precompute.py` locally to generate predictions.")


# ─── Architecture Info ───────────────────────────────────────────────────────
st.markdown('<div class="phase-header">🏗️ SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)

with st.expander("📐 Model Architecture Details", expanded=False):
    arch_col1, arch_col2 = st.columns(2)
    
    with arch_col1:
        st.markdown("""
        ### 🔮 FFNN (Feed-Forward Neural Network)
        ```
        ┌─────────────────────────┐
        │   Input Layer (12)      │  ← 12 monthly rainfall features
        ├─────────────────────────┤
        │   Dense(32)             │  ← Fully connected
        ├─────────────────────────┤
        │   Dense(16, ReLU)       │  ← Non-linear activation
        ├─────────────────────────┤
        │   Dense(1)              │  ← Water level prediction
        └─────────────────────────┘
        Optimizer: Adam | Loss: MSE
        ```
        """)
    
    with arch_col2:
        st.markdown("""
        ### 🧬 CNN2D (Convolutional Neural Network)
        ```
        ┌─────────────────────────┐
        │   Input (12×1×1)        │  ← Reshaped features
        ├─────────────────────────┤
        │   Conv2D(32, 3×1, ReLU) │  ← Feature extraction
        │   MaxPool(2×1)          │
        ├─────────────────────────┤
        │   Conv2D(64, 3×1, ReLU) │  ← Deep features
        │   MaxPool(2×1)          │
        ├─────────────────────────┤
        │   Flatten → Dense(100)  │  ← Classification
        │   Dense(1, Linear)      │  ← Prediction
        └─────────────────────────┘
        Optimizer: Adam | Loss: MSE
        ```
        """)
    
    st.markdown("""
    ### 🌐 Federated Learning Architecture
    - **3 Weather Stations** train models independently on local data
    - **Client** trains FFNN & CNN models, uploads to server
    - **Server** aggregates models using federated averaging
    - **Privacy**: Raw data never leaves the station
    """)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; padding: 20px;'>
    <p style='font-size: 0.9rem;'>
        🌊 <strong>FFM: Flood Forecasting Model</strong> — Powered by Federated Learning & Deep Neural Networks
    </p>
    <p style='font-size: 0.75rem;'>
        CNN2D + FFNN Architecture • MinMaxScaler Normalization • Real-time Prediction Engine
    </p>
    <p style='font-size: 0.7rem; margin-top: 10px;'>
        <a href="https://github.com/sivareddytalapareddy/floodforecasting" style="color: #3498db;">
            ⭐ GitHub Repository
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
# Cache bust
