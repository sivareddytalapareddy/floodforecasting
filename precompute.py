"""
Pre-compute all model predictions and save as JSON/NPZ files.
This allows the Streamlit app to run WITHOUT TensorFlow.
Run with: python3.10 precompute.py
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import math
import pickle

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'precomputed')
os.makedirs(OUT, exist_ok=True)

# Process all station datasets
datasets = ['Station_1.csv', 'Station_2.csv', 'Station_3.csv', 'FloodDataset.csv']

for ds_name in datasets:
    ds_path = os.path.join(BASE, 'Dataset', ds_name)
    if not os.path.exists(ds_path):
        print(f"Skipping {ds_name} - not found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {ds_name}")
    print(f"{'='*60}")
    
    dataset = pd.read_csv(ds_path)
    dataset.fillna(0, inplace=True)
    dataset_vals = dataset.values
    
    X = dataset_vals[:, 2:dataset_vals.shape[1]-1]
    Y = dataset_vals[:, dataset_vals.shape[1]-1]
    Y = Y.reshape(-1, 1)
    
    norm1 = MinMaxScaler(feature_range=(0, 1))
    norm2 = MinMaxScaler(feature_range=(0, 1))
    X = norm1.fit_transform(X)
    Y = norm2.fit_transform(Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    
    results = {
        'dataset_name': ds_name,
        'total_records': int(X.shape[0]),
        'total_features': int(X.shape[1]),
        'train_size': int(X_train.shape[0]),
        'test_size': int(X_test.shape[0]),
        'columns': dataset.columns.tolist(),
    }
    
    # Try years extraction
    try:
        years = dataset_vals[:, 1].flatten().astype(int).tolist()
        results['years'] = years
    except:
        results['years'] = list(range(1, len(dataset_vals) + 1))
    
    # FFNN Prediction
    ffnn_path = os.path.join(BASE, 'model', 'ff_weights.keras')
    if os.path.exists(ffnn_path):
        print("Loading FFNN model...")
        from keras.models import load_model
        model = load_model(ffnn_path)
        predict = model.predict(X_test)
        predict_inv = norm2.inverse_transform(np.abs(predict))
        test_inv = norm2.inverse_transform(y_test)
        
        r2 = r2_score(test_inv, predict_inv)
        mse_val = mean_squared_error(test_inv, predict_inv)
        rmse_val = math.sqrt(mse_val)
        acc = max(0, 100 - (rmse_val / np.mean(test_inv) * 100))
        
        results['ffnn'] = {
            'accuracy': float(acc),
            'mse': float(mse_val),
            'rmse': float(rmse_val),
            'r2': float(r2),
            'predictions': predict_inv.flatten().tolist(),
            'true_values': test_inv.flatten().tolist(),
        }
        print(f"  FFNN: Acc={acc:.2f}%, MSE={mse_val:.4f}, R²={r2:.4f}")
    
    # CNN Prediction
    cnn_path = os.path.join(BASE, 'model', 'extension_weights.keras')
    if os.path.exists(cnn_path):
        print("Loading CNN model...")
        from keras.models import load_model
        model_cnn = load_model(cnn_path)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
        predict_cnn = model_cnn.predict(X_test_cnn)
        predict_cnn_inv = norm2.inverse_transform(np.abs(predict_cnn))
        test_inv_cnn = norm2.inverse_transform(y_test)
        
        r2_c = r2_score(test_inv_cnn, predict_cnn_inv)
        mse_c = mean_squared_error(test_inv_cnn, predict_cnn_inv)
        rmse_c = math.sqrt(mse_c)
        acc_c = max(0, 100 - (rmse_c / np.mean(test_inv_cnn) * 100))
        
        results['cnn'] = {
            'accuracy': float(acc_c),
            'mse': float(mse_c),
            'rmse': float(rmse_c),
            'r2': float(r2_c),
            'predictions': predict_cnn_inv.flatten().tolist(),
            'true_values': test_inv_cnn.flatten().tolist(),
        }
        print(f"  CNN:  Acc={acc_c:.2f}%, MSE={mse_c:.4f}, R²={r2_c:.4f}")
    
    # Flood Forecast on test data
    test_data_path = os.path.join(BASE, 'Dataset', 'testData.csv')
    if os.path.exists(test_data_path) and os.path.exists(cnn_path):
        print("Running flood forecast on testData.csv...")
        test_df = pd.read_csv(test_data_path)
        test_df.fillna(0, inplace=True)
        test_vals = test_df.values
        
        features_start = 2
        features_end = test_vals.shape[1]
        
        try:
            expected = len(norm1.min_)
            current = features_end - features_start
            if current == expected + 1:
                features_end -= 1
        except:
            features_end -= 1
        
        X_forecast = test_vals[:, features_start:features_end]
        X_forecast = norm1.transform(X_forecast)
        X_forecast = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1, 1))
        
        forecast = model_cnn.predict(X_forecast)
        forecast = norm2.inverse_transform(forecast)
        
        try:
            forecast_years = test_vals[:, 1].flatten().astype(int).tolist()
        except:
            forecast_years = list(range(1, len(forecast) + 1))
        
        results['forecast'] = {
            'predictions': forecast.flatten().tolist(),
            'years': forecast_years,
            'source': 'testData.csv'
        }
        print(f"  Forecast: {len(forecast)} predictions generated")
    
    # Also forecast on test_now.csv
    test_now_path = os.path.join(BASE, 'Dataset', 'test_now.csv')
    if os.path.exists(test_now_path) and os.path.exists(cnn_path):
        print("Running flood forecast on test_now.csv...")
        test_now_df = pd.read_csv(test_now_path)
        test_now_df.fillna(0, inplace=True)
        test_now_vals = test_now_df.values
        
        features_start = 2
        features_end = test_now_vals.shape[1]
        try:
            expected = len(norm1.min_)
            current = features_end - features_start
            if current == expected + 1:
                features_end -= 1
        except:
            features_end -= 1
        
        X_fn = test_now_vals[:, features_start:features_end]
        X_fn = norm1.transform(X_fn)
        X_fn = np.reshape(X_fn, (X_fn.shape[0], X_fn.shape[1], 1, 1))
        
        forecast_now = model_cnn.predict(X_fn)
        forecast_now = norm2.inverse_transform(forecast_now)
        
        try:
            fn_years = test_now_vals[:, 1].flatten().astype(int).tolist()
        except:
            fn_years = list(range(1, len(forecast_now) + 1))
        
        results['forecast_now'] = {
            'predictions': forecast_now.flatten().tolist(),
            'years': fn_years,
            'source': 'test_now.csv'
        }
        print(f"  Forecast (now): {len(forecast_now)} predictions generated")
    
    # Save
    safe_name = ds_name.replace('.csv', '')
    out_path = os.path.join(OUT, f'{safe_name}_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

# Copy training history files
for hist_file in ['ff_history.json', 'cnn_history.json']:
    src = os.path.join(BASE, 'model', hist_file)
    dst = os.path.join(OUT, hist_file)
    if os.path.exists(src):
        import shutil
        shutil.copy2(src, dst)
        print(f"\nCopied: {hist_file}")

print("\n" + "="*60)
print("Pre-computation complete! All results saved to precomputed/")
print("="*60)
