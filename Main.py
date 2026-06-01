from tkinter import *
import tkinter
from tkinter import ttk
from PIL import Image, ImageTk
import time
from tkinter import filedialog
import numpy as np  # pyre-ignore[21]
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import socket
import json
import base64
from sklearn.metrics import mean_squared_error, r2_score
import math
from keras.layers import Dense, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.layers import  MaxPooling2D
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.callbacks import Callback


main = tkinter.Tk()
main.title("Flood Forecasting Client")
main.geometry("1100x700")
main.configure(bg='#f0f0f0')


# Initialize globals to prevent NameError
accuracy = []
mse = []
rmse = []
filename = ""
dataset = None
norm1 = None
norm2 = None
X = None
Y = None
X_train = None
y_train = None
X_test = None
y_test = None
active_dashboard = None
extension_model = None

# Load Assets Robustly using PIL
def load_icon(filename, size=(40, 40)):
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', filename)
        pil_image = Image.open(path)
        try:
            resample_filter = getattr(Image, 'Resampling').LANCZOS # type: ignore
        except AttributeError:
            resample_filter = getattr(Image, 'LANCZOS') # type: ignore
        pil_image = pil_image.resize(size, resample_filter)
        return ImageTk.PhotoImage(pil_image)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return PhotoImage(width=1, height=1)

try:
    # Resize to icon size (approx 40x40)
    icon_upload = load_icon("upload.png")
    icon_preprocess = load_icon("preprocess.png")
    icon_split = load_icon("split.png")
    icon_ffnn = load_icon("neural_net.png")
    icon_cnn = load_icon("cnn.png")
    icon_server = load_icon("server_upload.png")
    icon_chart = load_icon("chart.png")
    icon_predict = load_icon("predict.png")
    logo_img = load_icon("app_logo.png", (150, 150))
except Exception as e:
    print(f"Global Asset Error: {e}")



def show_splash():
    splash = Toplevel(main)
    splash.title("Loading...")
    splash.geometry("500x350")
    splash.overrideredirect(True) # No border
    
    # Center splash
    screen_width = main.winfo_screenwidth()
    screen_height = main.winfo_screenheight()
    x = (screen_width - 500) // 2
    y = (screen_height - 350) // 2
    splash.geometry(f"500x350+{x}+{y}")
    splash.configure(bg='white')
    
    Label(splash, image=logo_img, bg='white').pack(pady=20)
    Label(splash, text="Flood Forecasting System", font=("Helvetica", 18, "bold"), bg='white', fg='#333').pack()
    Label(splash, text="Initializing Neural Networks...", font=("Helvetica", 10), bg='white', fg='#666').pack(pady=5)
    
    style = ttk.Style()
    style.theme_use('default')
    style.configure("green.Horizontal.TProgressbar", background='#2ecc71')
    
    progress = ttk.Progressbar(splash, orient=HORIZONTAL, length=400, mode='determinate', style="green.Horizontal.TProgressbar")
    progress.pack(pady=20)
    
    # Simulated loading
    for i in range(101):
        progress['value'] = i
        splash.update_idletasks()
        time.sleep(0.015)
        
    splash.destroy()
    main.deiconify() # Show main window

# --- Live Training Dashboard ---
class LivePlotCallback(Callback):
    def __init__(self, update_queue):
        super().__init__()
        self.update_queue = update_queue
        self.history = {"loss": [], "val_loss": []}  # accumulate for JSON save

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(float(logs.get("loss", 0)))
        self.history["val_loss"].append(float(logs.get("val_loss", 0)))
        # Send a copy of logs to avoid concurrency issues
        self.update_queue.put((epoch, logs.copy()))
        time.sleep(0.01)

class TrainingDashboard(Toplevel):
    def __init__(self, parent, title="Training In Progress", total_epochs=100):
        super().__init__(parent)
        self.total_epochs = total_epochs
        self.title(title)
        self.geometry("600x600")
        self.configure(bg='#2b2b2b')
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6), dpi=90)
        self.fig.patch.set_facecolor('#2b2b2b')
        
        self.ax1.set_facecolor('#2b2b2b')
        self.ax1.set_title('Training & Validation Loss', color='white', fontsize=10)
        self.ax1.set_ylabel('Loss (MSE)', color='white')
        self.ax1.tick_params(colors='white')
        self.ax1.grid(color='white', alpha=0.1)
        
        self.ax2.set_facecolor('#2b2b2b')
        self.ax2.set_title('Training Progress', color='white', fontsize=10)
        self.ax2.set_xlabel('Epoch', color='white')
        self.ax2.tick_params(colors='white')
        self.ax2.grid(color='white', alpha=0.1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        self.epochs = []
        self.loss = []
        self.val_loss = []
        
        self.status_label = Label(self, text="Initializing Neural Network...", bg='#2b2b2b', fg='#00d4ff', font=("Consolas", 10))
        self.status_label.pack(side='bottom', fill='x', pady=5)
        
        # Stop button to force quit training if needed (Optional, but good for UX)
        # self.btn_stop = Button(self, text="STOP", bg="red", command=self.destroy)
        # self.btn_stop.pack(side='bottom')

    def update_plot(self, epoch, logs):
        self.epochs.append(epoch + 1)
        loss_val = logs.get('loss', 0)
        val_loss_val = logs.get('val_loss', 0)
        
        self.loss.append(loss_val)
        self.val_loss.append(val_loss_val)
        
        self.ax1.clear()
        self.ax1.set_facecolor('#2b2b2b')
        self.ax1.set_title('Loss Evolution (MSE)', color='white', fontsize=10)
        self.ax1.plot(self.epochs, self.loss, 'cyan', label='Train Loss', linewidth=1.5)
        self.ax1.plot(self.epochs, self.val_loss, 'orange', label='Val Loss', linewidth=1.5)
        self.ax1.legend(facecolor='#2b2b2b', labelcolor='white')
        self.ax1.grid(color='white', alpha=0.1)
        
        # Simplified Progress Bar effect on ax2
        self.ax2.clear()
        self.ax2.set_facecolor('#2b2b2b')
        self.ax2.barh([0], [epoch+1], color='#2ecc71', height=0.5)
        self.ax2.set_xlim(0, self.total_epochs)
        self.ax2.set_yticks([])
        self.ax2.set_title(f"Epoch {epoch+1} / {self.total_epochs}", color='white')
        
        self.canvas.draw()
        self.status_label.config(text=f"Epoch {epoch+1} | Loss: {loss_val:.5f} | Val Loss: {val_loss_val:.5f}")

    def reset_data(self):
        self.epochs = []
        self.loss = []
        self.val_loss = []
        self.ax1.clear()
        self.ax2.clear()
        self.status_label.config(text="Resetting for new training run...")
        self.canvas.draw()

    def plot_prediction(self, y_true, y_pred, algorithm_name="Model",
                        was_cached=False, history_json=None):
        """Plot prediction on ax2; restore real MSE loss curve on ax1 if available."""
        import numpy as np
        import json as _json
        from sklearn.metrics import mean_squared_error
        import math

        # ── ax1: Real loss curve (from JSON) or cache notice ───────────────────────
        hist_loaded = False
        if history_json and os.path.exists(history_json):
            try:
                with open(history_json, "r") as f:
                    saved = _json.load(f)
                h_loss     = saved.get("loss", [])
                h_val_loss = saved.get("val_loss", [])
                if h_loss:
                    epochs_range = list(range(1, len(h_loss) + 1))
                    self.ax1.clear()
                    self.ax1.set_facecolor('#2b2b2b')
                    self.ax1.set_title('Training & Validation Loss (MSE)', color='white', fontsize=10)
                    self.ax1.plot(epochs_range, h_loss,     color='cyan',   label='Train Loss', linewidth=1.5)
                    self.ax1.plot(epochs_range, h_val_loss, color='orange', label='Val Loss',   linewidth=1.5)
                    self.ax1.set_xlabel('Epoch', color='white', fontsize=8)
                    self.ax1.set_ylabel('Loss (MSE)', color='white', fontsize=8)
                    self.ax1.legend(facecolor='#2b2b2b', labelcolor='white', fontsize=8)
                    self.ax1.tick_params(colors='white')
                    self.ax1.grid(color='white', alpha=0.1)
                    hist_loaded = True
            except Exception:
                pass  # fall through to 'no history' notice

        if not hist_loaded and (was_cached or len(self.loss) == 0):
            self.ax1.clear()
            self.ax1.set_facecolor('#2b2b2b')
            self.ax1.set_title('Training Loss (MSE)', color='white', fontsize=10)
            self.ax1.text(
                0.5, 0.5,
                f"{algorithm_name}\nModel loaded from cache\nDelete .keras file to retrain",
                color='#00d4ff', ha='center', va='center',
                fontsize=10, transform=self.ax1.transAxes
            )
            self.ax1.tick_params(colors='white')
            self.ax1.grid(color='white', alpha=0.1)
        elif not hist_loaded:
            # Fresh training — loss curve already plotted by update_plot(); leave it.
            pass

        # ── ax2: Prediction vs True ───────────────────────────────────────────
        self.ax2.clear()
        self.ax2.set_facecolor('#2b2b2b')
        self.ax2.set_title(f'{algorithm_name} Prediction vs True Water Level', color='white', fontsize=10)

        self.ax2.plot(y_true,  color='#e74c3c', label='True',      linewidth=1.5)
        self.ax2.plot(y_pred,  color='#2ecc71', label='Predicted', linewidth=1.5)

        # Annotate MSE value on the prediction plot x-label
        try:
            mse_val  = mean_squared_error(y_true, y_pred)
            rmse_val = math.sqrt(mse_val)
            self.ax2.set_xlabel(
                f'Sample Index   [MSE: {mse_val:.4f}  |  RMSE: {rmse_val:.4f}]',
                color='#aaaaaa', fontsize=8
            )
        except Exception:
            self.ax2.set_xlabel('Sample Index', color='white')

        self.ax2.set_ylabel('Water Level', color='white')
        self.ax2.legend(facecolor='#2b2b2b', labelcolor='white')
        self.ax2.grid(color='white', alpha=0.1)
        self.ax2.tick_params(colors='white')

        self.canvas.draw()
        self.status_label.config(text=f"Training & Evaluation Completed for {algorithm_name}")



main.withdraw() # Hide main initially
main.after(100, show_splash)  # type: ignore



def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    text.insert(END,str(dataset))

def preprocessDataset():
    global filename, dataset, norm1, norm2, X, Y
    text.delete('1.0', END)
    norm1 = MinMaxScaler(feature_range = (0, 1))
    norm2 = MinMaxScaler(feature_range = (0, 1))
    dataset_vals = dataset.values # type: ignore
    X = dataset_vals[:,2:dataset_vals.shape[1]-1] # type: ignore
    Y = dataset_vals[:,dataset_vals.shape[1]-1] # type: ignore
    
    # CRITICAL FIX: DO NOT SHUFFLE TIME SERIES DATA for CNN/LSTM
    # indices = np.arange(X.shape[0])
    # np.random.shuffle(indices)
    # X = X[indices]
    # Y = Y[indices]
    
    Y = Y.reshape(-1, 1)
    X  = norm1.fit_transform(X)
    Y = norm2.fit_transform(Y)
    text.insert(END,"Dataset preprocessing (Normalization) Completed.\nShuffling Disabled to preserve Time-Series patterns.\n\n")
    text.insert(END,"Normalized Dataset\n\n")
    text.insert(END,str(X))

def datasetSplit():
    global X, Y, X_train, y_train, X_test, y_test
    text.delete('1.0', END)
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    # CRITICAL FIX: shuffle=False to keep historical order
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle=False)
    text.insert(END,"Total records found in dataset  = "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset = "+str(X.shape[1])+"\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing  : "+str(X_test.shape[0])+"\n")

#function to calculate MSE and other metrics
def calculateMetrics(algorithm, predict, test_labels, dashboard=None, was_cached=False, history_json=None):
    predict = norm2.inverse_transform(np.abs(predict)) # type: ignore
    test_label = norm2.inverse_transform(test_labels) # type: ignore
    r2 = r2_score(test_label, predict)
    mse_value = mean_squared_error(test_label, predict)
    rmse_value = math.sqrt(mse_value)
    
    # Robust Accuracy: 100 - MAPE/NormalizedError
    # Simple proxy: 100 - (RMSE / Mean * 100)
    acc = 100 - (rmse_value / np.mean(test_label) * 100)
    if acc < 0: acc = 0
    
    mse.append(mse_value)
    rmse.append(rmse_value)
    accuracy.append(acc)
    text.insert(END,algorithm+" MSE      : "+str(mse_value)+"\n")
    text.insert(END,algorithm+" RMSE     : "+str(rmse_value)+"\n")
    text.insert(END,algorithm+" Accuracy : "+str(acc)+"\n") 
    text.insert(END,algorithm+" R2 SCORE : "+str(r2)+"\n\n")    
    for i in range(len(predict)):
        text.insert(END,"True Water Level : "+str(test_label[i])+" Predicted Water Level : "+str(predict[i])+"\n")
    
    if dashboard:
        dashboard.plot_prediction(test_label, predict, algorithm,
                                  was_cached=was_cached, history_json=history_json)
    else:
        # Legacy: New popup
        plt.figure()
        plt.plot(test_label, color = 'red', label = 'True Water Level')
        plt.plot(predict, color = 'green', label = 'Predicted Water Level')
        plt.title(algorithm+' Water Level Prediction')
        plt.xlabel('Test Data')
        plt.ylabel('Predicted Water Level')
        plt.legend()
        plt.show()    

def runFFNN():
    global X, Y, X_train, y_train, X_test, y_test
    global accuracy, mse, rmse
    
    if 'X_train' not in globals():
        from tkinter import messagebox
        messagebox.showerror("Error", "Please Split Dataset first.")
        return

    text.delete('1.0', END)
    accuracy = []
    mse = []
    rmse = []
    
    # 1. Setup Dashboard & Queue (Singleton)
    global active_dashboard
    if 'active_dashboard' in globals() and active_dashboard is not None:
        try:
            if active_dashboard.winfo_exists():
                active_dashboard.title("Training FFNN (Baseline)") # Reuse window
                active_dashboard.total_epochs = 100
                dash = active_dashboard
                dash.reset_data() # Clear old graphs
            else:
                dash = TrainingDashboard(main, title="Training FFNN (Baseline)", total_epochs=100)
                active_dashboard = dash
        except:
             dash = TrainingDashboard(main, title="Training FFNN (Baseline)", total_epochs=100)
             active_dashboard = dash
    else:
        dash = TrainingDashboard(main, title="Training FFNN (Baseline)", total_epochs=100)
        active_dashboard = dash
        
    update_q = queue.Queue()
    
    # 2. Worker Thread
    def train_worker():
        try:
            # ── Cached-weights check ──────────────────────────────────────────
            # If a previously saved model exists, skip retraining entirely.
            # Delete 'model/ff_weights.keras' manually to force a fresh train.
            if os.path.exists('model/ff_weights.keras'):
                update_q.put("LOADED")
                return

            # Re-create model structure inside thread
            model = Sequential()
            model.add(Dense(32,  input_shape=(X.shape[1],))) # type: ignore
            model.add(Dense(16, activation="relu"))
            model.add(Dense(units=1))
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            
            model_check_point = ModelCheckpoint(filepath='model/ff_weights.keras', verbose = 0, save_best_only = True)
            cb = LivePlotCallback(update_q)
            model.fit(X_train, y_train, epochs = 100, batch_size = 8, 
                      validation_data=(X_test, y_test), 
                      callbacks=[model_check_point, cb], 
                      verbose=0)
            # Save training history for cache reload
            try:
                import json as _j
                with open('model/ff_history.json', 'w') as f:
                    _j.dump(cb.history, f)
            except Exception:
                pass
            update_q.put("DONE")
                
        except Exception as e:
            update_q.put(f"ERROR:{str(e)}")

    # 3. Start Thread
    t = threading.Thread(target=train_worker)
    t.daemon = True
    t.start()
    
    # 4. Monitor Loop (Main Thread)
    def monitor_training():
        try:
            # Drain queue
            while True:
                item = update_q.get_nowait()
                if item == "DONE":
                    # dash.destroy() # Don't close
                    finalize_training(False)
                    return
                elif item == "LOADED":
                    # dash.destroy() # Don't close
                    finalize_training(True)
                    return
                elif isinstance(item, str) and item.startswith("ERROR"):
                    dash.destroy()
                    text.insert(END, item + "\n")
                    return
                else:
                    # (epoch, logs)
                    dash.update_plot(item[0], item[1])
        except queue.Empty:
            pass
        
        if t.is_alive():
            main.after(50, monitor_training)  # type: ignore
        else:
            # Edge case: Thread died but queue empty? 
            # Could check once more.
            pass

    # 5. Post-Training Logic
    def finalize_training(was_loaded):
        if was_loaded:
            text.insert(END, "Model found on disk. Loading without retraining...\n")
        else:
            text.insert(END, "Training Completed Successfully.\n")
            
        try:
            model = load_model('model/ff_weights.keras')
            predict = model.predict(X_test)
            calculateMetrics("FFNN", predict, y_test, dashboard=dash,
                             was_cached=was_loaded, history_json='model/ff_history.json')
        except Exception as e:
            text.insert(END, f"Error loading model: {e}\n")

    monitor_training()

def runExtension():
    global X, Y, X_train, y_train, X_test, y_test
    global accuracy, mse, rmse, extension_model
    
    if 'X_train' not in globals():
        from tkinter import messagebox
        messagebox.showerror("Error", "Please Split Dataset first.")
        return

    text.delete('1.0', END)
    # Note: We do NOT reset accuracy/mse lists here, to allow comparison with FFNN
    
    # 1. Setup Dashboard (Singleton)
    global active_dashboard
    if 'active_dashboard' in globals() and active_dashboard is not None:
        try:
            if active_dashboard.winfo_exists():
                active_dashboard.title("Training CNN Extension (Deep Learning)")
                active_dashboard.total_epochs = 150
                dash = active_dashboard
                dash.reset_data() # Clear old graphs
            else:
                dash = TrainingDashboard(main, title="Training CNN Extension (Deep Learning)", total_epochs=150)
                active_dashboard = dash
        except:
             dash = TrainingDashboard(main, title="Training CNN Extension (Deep Learning)", total_epochs=150)
             active_dashboard = dash
    else:
        dash = TrainingDashboard(main, title="Training CNN Extension (Deep Learning)", total_epochs=150)
        active_dashboard = dash
        
    update_q = queue.Queue()
    
    # 2. Worker Thread
    def train_worker():
        try:
            # ── Cached-weights check ──────────────────────────────────────────
            # If a previously saved model exists, skip retraining entirely.
            # Delete 'model/extension_weights.keras' manually to force a fresh train.
            if os.path.exists('model/extension_weights.keras'):
                update_q.put("LOADED")
                return

            # Reshape locally for 2D Conv
            if X_train is None: raise ValueError("No training data")
            
            X_train1 = X_train.reshape(X_train.shape[0],X_train.shape[1], 1, 1) # type: ignore
            X_test1 = X_test.reshape(X_test.shape[0],X_test.shape[1], 1, 1) # type: ignore
            
            ext_model = Sequential()
            ext_model.add(Convolution2D(32, (3, 1), input_shape = (12, 1, 1), activation = 'relu', padding='same'))
            ext_model.add(MaxPooling2D(pool_size = (2, 1)))
            ext_model.add(Convolution2D(64, (3, 1), activation = 'relu', padding='same'))
            ext_model.add(MaxPooling2D(pool_size = (2, 1)))
            
            ext_model.add(Flatten())
            ext_model.add(Dense(100, activation = 'relu'))
            ext_model.add(Dense(1, activation='linear')) 
            
            ext_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            
            cb = LivePlotCallback(update_q)
            model_check_point = ModelCheckpoint(filepath='model/extension_weights.keras', verbose = 0, save_best_only = True)
            ext_model.fit(X_train1, y_train, epochs = 150, batch_size = 8, 
                          validation_data=(X_test1, y_test), 
                          callbacks=[model_check_point, cb], 
                          verbose=0)
            # Save training history for cache reload
            try:
                import json as _j
                with open('model/cnn_history.json', 'w') as f:
                    _j.dump(cb.history, f)
            except Exception:
                pass
            update_q.put("DONE")
        except Exception as e:
            update_q.put(f"ERROR:{str(e)}")

    # 3. Start Thread
    t = threading.Thread(target=train_worker)
    t.daemon = True
    t.start()
    
    # 4. Monitor
    def monitor_training():
        try:
            while True:
                item = update_q.get_nowait()
                if item == "DONE":
                    # dash.destroy()
                    finalize_training(False)
                    return
                elif item == "LOADED":
                    # dash.destroy()
                    finalize_training(True)
                    return
                elif isinstance(item, str) and item.startswith("ERROR"):
                    dash.destroy()
                    text.insert(END, item + "\n")
                    return
                else:
                    dash.update_plot(item[0], item[1])
        except queue.Empty:
            pass
            
        if t.is_alive():
            main.after(50, monitor_training)  # type: ignore
        else:
            pass

    # 5. Finalize
    def finalize_training(was_loaded):
        global extension_model
        if was_loaded:
             text.insert(END, "CNN Model found using existing weights.\n")
        else:
             text.insert(END, "CNN Training Completed.\n")
        
        try:
            X_test1 = X_test.reshape(X_test.shape[0],X_test.shape[1], 1, 1)
            extension_model = load_model('model/extension_weights.keras')
            predict = extension_model.predict(X_test1)
            calculateMetrics("Extension CNN2D", predict, y_test, dashboard=dash,
                             was_cached=was_loaded, history_json='model/cnn_history.json')
        except Exception as e:
            text.insert(END, f"Error during evaluation: {e}\n")

    monitor_training()

def uploadtoServer():
    text.delete('1.0', END)
    station_name = simpledialog.askstring("Station Name", "Enter Station Name (e.g., Station_1, Station_2, Station_3):", parent=main, initialvalue="Station_1")
    if not station_name: return

    # Helper function to send model
    def send_model(model_type, file_path):
        if not os.path.exists(file_path):
            text.insert(END, f"Skipping {model_type}: File not found ({file_path})\n")
            messagebox.showwarning("Missing Model", f"Could not find trained {model_type} model.\nPlease run '{model_type if model_type=='FFNN' else 'Extension CNN'}' training first.")
            return
            
        try:
            with open(file_path, 'rb') as file:
                model_data = base64.b64encode(file.read()).decode()
            
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
            client.connect(('localhost', 2222))
            
            payload = {
                "request": 'update_model', 
                "station": station_name, 
                "model_type": model_type,
                "model": model_data
            }
            
            jsondata = json.dumps(payload)
            # Add delimiter for robust server-side reading
            client.send((jsondata + "<EOF>").encode())
            
            # Simple wait for response (blocks until server replies or closes)
            data = client.recv(1024).decode()
            client.close()
            text.insert(END, f"{model_type} Upload: {data}\n")
        except Exception as e:
            text.insert(END, f"{model_type} Upload Error: {str(e)}\n")

    # Upload FFNN
    send_model("FFNN", 'model/ff_weights.keras')
    
    time.sleep(1.0) # Prevent socket overlap
    
    # Upload CNN
    send_model("CNN", 'model/extension_weights.keras')
    
    text.insert(END, "\nUpload Process Completed.\nCheck Server Dashboard.\n\n")

from tkinter import messagebox

def graph():
    # Robust check for data existence
    if 'mse' not in globals() or len(mse) < 2:
        messagebox.showerror("Graph Error", "Insufficient data for comparison.\n\nPlease run BOTH algorithms (FFNN and Extension CNN) first.")
        return

    try:
        df = pd.DataFrame([['Propose FFNN','MSE',mse[0]],['Propose FFNN','RMSE',rmse[0]], ['Propose FFNN','Accuracy',accuracy[0]],
                           ['Extension CNN2D','MSE',mse[1]],['Extension CNN2D','RMSE',rmse[1]],['Extension CNN2D','Accuracy',accuracy[1]],                       
                      ],columns=['Parameters','Algorithms','Value'])
        df.pivot(index="Parameters", columns="Algorithms", values="Value").plot(kind='bar')
        plt.title("Propose FFNN & Extension CNN2D Performance Graph")
        plt.show()
    except Exception as e:
        messagebox.showerror("Plot Error", f"An error occurred while plotting:\n{str(e)}")

def predict():
    global norm1, norm2, extension_model
    try:
        filename = filedialog.askopenfilename(initialdir = "Dataset")
        if not filename: return
        
        # read test data and predict flood
        dataset = pd.read_csv(filename)
        dataset.fillna(0, inplace = True)
        dataset_values = dataset.values
        
        # Check if we need to drop the last column (Target) to match training features (12)
        # Training used: X = dataset_vals[:,2:dataset_vals.shape[1]-1]
        
        # If the scaler expects N features, and our slice gives N+1, we assume the last one is Target.
        features_start = 2
        features_end = dataset_values.shape[1]
        
        # Try to infer expected features from scaler if possible, else default to -1 logic if file looks standard
        try:
             # Scikit-learn < 0.24 uses .scale_ or .min_ to infer n_features
            expected_features = len(norm1.min_) # type: ignore
            current_features = features_end - features_start
            
            if current_features == expected_features + 1:
                features_end = features_end - 1 # Exclude target
        except:
             # Fallback if norm1 isn't ready or older sklearn version behaving differently
             features_end = features_end - 1

        X = dataset_values[:, features_start:features_end]
        X = norm1.transform(X) # type: ignore
        X = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
        
        predict_output = extension_model.predict(X)  # type: ignore
        predict_output = norm2.inverse_transform(predict_output) # type: ignore
        
        text.delete('1.0', END)
        text.insert(END, f"Forecast Results for {os.path.basename(filename)}\n")
        text.insert(END, "-"*50 + "\n")
        
        # Threshold for Flood Alert (e.g., above 25 units)
        FLOOD_THRESHOLD = 25.0 
        flood_detected = False
        
        # Try to extract YEAR column (Index 1) for visualization
        years = []
        try:
            # Assuming standard dataset format: SUBDIVISION, YEAR, JAN... 
            # YEAR is usually at index 1
            if len(dataset.columns) > 1 and 'YEAR' in dataset.columns[1].upper() or isinstance(dataset_values[0][1], (int, float)):
                 years = dataset_values[:, 1].flatten().astype(int)
        except:
             pass
             
        # Generate generic years if extraction failed
        if len(years) != len(predict_output):
             years = list(range(1, len(predict_output) + 1))
             time_label = "Time Step"
        else:
             time_label = "Year"

        for i in range(len(predict_output)):
            val = predict_output[i,0]
            status = "NORMAL"
            if val > FLOOD_THRESHOLD:
                status = "CRITICAL"
                flood_detected = True
            
            # Use Year if available
            time_str = f"{time_label} {years[i]}"
            text.insert(END, f"{time_str}: Water Level = {val:.2f}  [{status}]\n")
            
        text.insert(END, "-"*50 + "\n")
        
        # 1. Flood Alert Popup
        if flood_detected:
            messagebox.showwarning("CRITICAL FLOOD ALERT", 
                                   f"WARNING: Predicted water levels exceed critical threshold ({FLOOD_THRESHOLD})!\n\n"
                                   "Immediate attention required.\n\n"
                                   "Click OK to view the Forecast Trend Graph.")
        else:
            messagebox.showinfo("Forecast Status", "Forecast complete. Water levels are within normal range.\n\nClick OK to view the Forecast Trend Graph.")

        # 2. Trend Graph (Mesmerizing UI)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Data
        ax.plot(years, predict_output, color='#2980b9', linewidth=3, marker='o', markersize=8, label='Predicted Water Level')
        
        # Danger Zones (Red above threshold, Green below)
        ax.axhspan(FLOOD_THRESHOLD, max(predict_output.max(), FLOOD_THRESHOLD + 10), color='#e74c3c', alpha=0.3, label='Danger Zone') # type: ignore
        ax.axhspan(min(predict_output.min(), 0), FLOOD_THRESHOLD, color='#2ecc71', alpha=0.3, label='Safe Zone') # type: ignore
        
        # Threshold Line
        ax.axhline(y=FLOOD_THRESHOLD, color='#c0392b', linestyle='--', linewidth=2, label='Flood Threshold')
        
        # Peak Annotation
        max_val = predict_output.max()
        max_idx = np.argmax(predict_output)
        max_year = years[max_idx]
        
        if max_val > FLOOD_THRESHOLD:
            ax.annotate(f'PEAK FLOOD\n{max_val:.1f} units', 
                        xy=(max_year, max_val), 
                        xytext=(max_year, max_val + 5),
                        arrowprops={'facecolor': 'black', 'shrink': 0.05},
                        fontsize=12, fontweight='bold', color='red', ha='center')

        # Styling
        ax.set_title(f"Flood Forecast Analysis: {os.path.basename(filename)}", fontsize=16, fontweight='bold', color='#2c3e50')
        ax.set_xlabel(time_label, fontsize=12)
        ax.set_ylabel("Water Level (Units)", fontsize=12)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        if time_label == "Year":
            plt.xticks(years, rotation=45) 
            
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        messagebox.showerror("Forecast Error", f"An error occurred during forecasting:\n{str(e)}")



       
font = ('Segoe UI', 14, 'bold')
title_frame = Frame(main, bg='#ecf0f1', pady=20)
title_frame.pack(side=TOP, fill='x') # type: ignore

# Load logo for title if available
try:
    # Re-use logo_img from global scope
    if 'logo_img' in globals():
        logo_label = Label(title_frame, image=logo_img, bg='#ecf0f1')
        logo_label.pack(side=LEFT, padx=20)
except:
    pass

title = Label(title_frame, text='FFM: Flood Forecasting Model Using Federated Learning', justify=CENTER)
title.config(bg='#ecf0f1', fg='#2c3e50', font=font)           
title.pack(side=LEFT, padx=10)

# Main Container
main_container = Frame(main, bg='#f4f6f7')
main_container.pack(fill=BOTH, expand=True, padx=20, pady=20)

# Left Panel Container (Holds Canvas + Scrollbar)
left_container = Frame(main_container, bg='#f4f6f7', width=420) # Increased width slightly for scrollbar
left_container.pack(side=LEFT, fill='y', padx=(0, 20)) # type: ignore
left_container.pack_propagate(False) # Force width respect

# Canvas for Scrolling
left_canvas = Canvas(left_container, bg='#f4f6f7', highlightthickness=0)
left_scrollbar = Scrollbar(left_container, orient="vertical", command=left_canvas.yview)

# Scrollable Frame (This becomes the technical 'left_panel' for button parentage)
left_panel = Frame(left_canvas, bg='#f4f6f7')

# Configure Scroll Region
def on_frame_configure(event):
    left_canvas.configure(scrollregion=left_canvas.bbox("all"))

left_panel.bind("<Configure>", on_frame_configure)

# Add Window to Canvas
left_canvas_window = left_canvas.create_window((0, 0), window=left_panel, anchor="nw")

def on_canvas_configure(event):
    # Resize inner frame to match canvas width
    canvas_width = event.width
    left_canvas.itemconfig(left_canvas_window, width=canvas_width)

left_canvas.bind("<Configure>", on_canvas_configure)

left_canvas.configure(yscrollcommand=left_scrollbar.set)

left_canvas.pack(side=LEFT, fill='both', expand=True) # type: ignore
left_scrollbar.pack(side=RIGHT, fill='y') # type: ignore

# Bind MouseWheel to specific canvas
def _on_mousewheel(event):
    left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
# Bind enter/leave events to enable/disable mousewheel for this area
def _bound_to_mousewheel(event):
    left_canvas.bind_all("<MouseWheel>", _on_mousewheel) # type: ignore

def _unbound_to_mousewheel(event):
    left_canvas.unbind_all("<MouseWheel>") # type: ignore

left_panel.bind('<Enter>', _bound_to_mousewheel)
left_panel.bind('<Leave>', _unbound_to_mousewheel)

# Right Panel (Logs)
right_panel = Frame(main_container, bg='white', bd=1, relief=SOLID)
right_panel.pack(side=RIGHT, fill='both', expand=True) # type: ignore

# --- Phase Styles ---
phase_font = ('Segoe UI', 10, 'bold')
lbl_bg = '#f4f6f7'

def create_phase_frame(parent, text):
    frame = LabelFrame(parent, text=text, font=phase_font, bg=lbl_bg, fg='#34495e', bd=2, relief=GROOVE)
    frame.pack(fill='x', pady=5, ipady=5) # type: ignore
    return frame

# --- Phase 1: Data Pipeline ---
p1_frame = create_phase_frame(left_panel, "PHASE 1: DATA PIPELINE")

# Button Styles (Refined)
# Removed fixed height to prevent clipping; added internal padding

# Helper to create buttons
def add_btn(parent, text, cmd, icon, bg_color):
    btn = Button(parent, text=text, command=cmd, image=icon, bg=bg_color, fg='white', activebackground=bg_color, activeforeground='white', font=('Segoe UI', 11, 'bold'), bd=0, cursor='hand2', compound=LEFT, anchor='w', padx=20, pady=12)
    # Added ipady for extra click area and visual breathing room
    btn.pack(fill='x', padx=15, pady=8) # type: ignore
    return btn

add_btn(p1_frame, "1. Upload CSV Dataset", uploadDataset, icon_upload, '#3498db')
add_btn(p1_frame, "2. Preprocess Data", preprocessDataset, icon_preprocess, '#1abc9c')
add_btn(p1_frame, "3. Train / Test Split", datasetSplit, icon_split, '#e67e22')

# --- Phase 2: Model Training ---
p2_frame = create_phase_frame(left_panel, "PHASE 2: MODEL TRAINING")
add_btn(p2_frame, "4. Train FFNN (Baseline)", runFFNN, icon_ffnn, '#9b59b6')
add_btn(p2_frame, "5. Train CNN2D (Extension)", runExtension, icon_cnn, '#34495e')

# --- Phase 3: Federated Learning ---
p3_frame = create_phase_frame(left_panel, "PHASE 3: FEDERATED LEARNING")
add_btn(p3_frame, "6. Upload to Server", uploadtoServer, icon_server, '#00bcd4')

# --- Phase 4: Prediction & Analytics ---
p4_frame = create_phase_frame(left_panel, "PHASE 4: PREDICTION & ANALYTICS")
add_btn(p4_frame, "7. Accuracy Graph", graph, icon_chart, '#e91e63')
add_btn(p4_frame, "8. Flood Forecast", predict, icon_predict, '#f1c40f')

# --- Phase 5: Simulation (Removed) ---
# p5_frame = create_phase_frame(left_panel, "PHASE 5: SIMULATION & STRESS TEST")
# add_btn(p5_frame, "9. God Mode (Simulation)", runSimulation, icon_chart, '#34495e')

# Console Output
lbl_console = Label(right_panel, text="System Logs & Output", font=('Segoe UI', 10, 'bold'), bg='white', fg='#7f8c8d', anchor='w')
lbl_console.pack(fill='x', padx=10, pady=(10, 5)) # type: ignore

font_txt = ('Consolas', 10)
text = Text(right_panel, bd=0, padx=10, pady=10, state=NORMAL)
scroll = Scrollbar(right_panel, command=text.yview)
text.configure(yscrollcommand=scroll.set, bg='#fdfefe', fg='#2c3e50', font=font_txt)

scroll.pack(side=RIGHT, fill=Y)
text.pack(side=LEFT, fill=BOTH, expand=True)

main.mainloop()
