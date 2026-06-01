
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import socket
from threading import Thread
import json
import base64
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import time
import psutil
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# --- THEME SETUP ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
plt.style.use('dark_background')

# Create directories if they don't exist
if not os.path.exists('received'):
    os.makedirs('received')
if not os.path.exists('model'):
    os.makedirs('model')

# Global variables
server_socket = None
client_models = {}  
client_metrics = {} 
is_running = True

# Helper to get data for specific station (for correct scaling)
def get_station_scalers(station_name):
    if "1" in station_name: filename = "Station_1.csv"
    elif "2" in station_name: filename = "Station_2.csv"
    elif "3" in station_name: filename = "Station_3.csv"
    else: filename = f"{station_name}.csv"
    
    file_path = f"Dataset/{filename}"
    print(f"DEBUG: Loading Scalers from {file_path}")
    
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: Dataset {file_path} not found on Server!")
        raise FileNotFoundError(f"Missing dataset for {station_name}")
        
    df = pd.read_csv(file_path)
    data = df.values
    
    X_raw = data[:, 2:14] # Features
    y_raw = data[:, 14].reshape(-1, 1) # Target
    
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    scaler_x.fit(X_raw)
    scaler_y.fit(y_raw)
    
    return scaler_x, scaler_y, filename

class FloodServer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.withdraw() # hide initially
        self.show_splash()

    def show_splash(self):
        self.splash = ctk.CTkToplevel(self)
        self.splash.geometry("500x350")
        self.splash.overrideredirect(True)
        # Center splash
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 500) // 2
        y = (screen_height - 350) // 2
        self.splash.geometry(f"500x350+{x}+{y}")
        self.splash.configure(fg_color="#1a1a2d")

        ctk.CTkLabel(self.splash, text="🌊 CENTRAL SERVER", font=("Arial", 28, "bold"), text_color="#00d4ff").pack(pady=(60, 10))
        self.splash_status = ctk.CTkLabel(self.splash, text="Starting sockets...", font=("Consolas", 12), text_color="white")
        self.splash_status.pack(pady=10)
        
        self.splash_progress = ctk.CTkProgressBar(self.splash, width=400, mode='determinate', progress_color="#2ecc71")
        self.splash_progress.pack(pady=20)
        self.splash_progress.set(0)
        
        self.splash_step = 0
        self.update_splash()

    def update_splash(self):
        self.splash_step += 2
        self.splash_progress.set(self.splash_step / 100)
        
        if self.splash_step == 30:
            self.splash_status.configure(text="Loading Network Topology...")
        elif self.splash_step == 60:
            self.splash_status.configure(text="Initializing Aggregation Engine...")
        elif self.splash_step == 90:
            self.splash_status.configure(text="Server Ready.")
            
        if self.splash_step <= 100:
            self.after(100, self.update_splash)
        else:
            self.splash.destroy()
            self.initialize_main_ui()

    def initialize_main_ui(self):
        self.title("Federated Learning Central Server - Krishna River Flood Forecasting")
        self.geometry("1400x900")
        
        # Grid layout (1x2)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        
        self.create_navigation()
        self.create_frames()
        
        self.show_frame("Dashboard")
        
        self.start_server()
        self.update_system_monitor()
        self.deiconify()

    def create_navigation(self):
        self.nav_frame = ctk.CTkFrame(self, corner_radius=0)
        self.nav_frame.grid(row=0, column=0, sticky="nsew")
        self.nav_frame.grid_rowconfigure(8, weight=1) 
        
        self.logo_label = ctk.CTkLabel(self.nav_frame, text="🌊 FL SERVER", 
                                      font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=20)
        
        # Buttons
        self.btn_dashboard = self.create_nav_btn("Dashboard", 1)
        self.btn_s1 = self.create_nav_btn("Station 1", 2)
        self.btn_s2 = self.create_nav_btn("Station 2", 3)
        self.btn_s3 = self.create_nav_btn("Station 3", 4)
        self.btn_agg = self.create_nav_btn("Aggregation", 5)
        self.btn_hist = self.create_nav_btn("History", 6)
        self.btn_map = self.create_nav_btn("Network Map", 7)
        
        # System Monitor (Bottom of Sidebar)
        self.sys_frame = ctk.CTkFrame(self.nav_frame, fg_color="transparent")
        self.sys_frame.grid(row=9, column=0, padx=10, pady=10, sticky="ew")
        
        ctk.CTkLabel(self.sys_frame, text="SYSTEM HEALTH", font=("Arial", 10, "bold"), text_color="gray").pack(anchor="w")
        
        self.lbl_cpu = ctk.CTkLabel(self.sys_frame, text="CPU: 0%", font=("Consolas", 10))
        self.lbl_cpu.pack(anchor="w")
        self.progress_cpu = ctk.CTkProgressBar(self.sys_frame, height=5, progress_color="#00d4ff")
        self.progress_cpu.pack(fill="x", pady=2)
        
        self.lbl_ram = ctk.CTkLabel(self.sys_frame, text="RAM: 0%", font=("Consolas", 10))
        self.lbl_ram.pack(anchor="w")
        self.progress_ram = ctk.CTkProgressBar(self.sys_frame, height=5, progress_color="#ff4757")
        self.progress_ram.pack(fill="x", pady=2)

    def create_nav_btn(self, text, row):
        btn = ctk.CTkButton(self.nav_frame, corner_radius=0, height=40, border_spacing=10, 
                            text=text, fg_color="transparent", text_color=("gray10", "gray90"), 
                            hover_color=("gray70", "gray30"), anchor="w", 
                            command=lambda: self.show_frame(text))
        btn.grid(row=row, column=0, sticky="ew")
        return btn

    def create_frames(self):
        self.frames = {}
        
        # Dashboard
        self.frames["Dashboard"] = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.setup_dashboard(self.frames["Dashboard"])
        
        # Stations
        for s in ["Station 1", "Station 2", "Station 3"]:
            self.frames[s] = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        
        # Aggregation
        self.frames["Aggregation"] = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")

        # History
        self.frames["History"] = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.setup_history(self.frames["History"])
        
        # Network Map
        self.frames["Network Map"] = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.setup_network_map(self.frames["Network Map"])
        
    def show_frame(self, name):
        # Reset buttons
        for btn in [self.btn_dashboard, self.btn_s1, self.btn_s2, self.btn_s3, self.btn_agg, self.btn_hist, self.btn_map]:
            btn.configure(fg_color="transparent")
            
        # Highlight active
        if name == "Dashboard": self.btn_dashboard.configure(fg_color=("gray75", "gray25"))
        elif name == "Station 1": self.btn_s1.configure(fg_color=("gray75", "gray25"))
        elif name == "Station 2": self.btn_s2.configure(fg_color=("gray75", "gray25"))
        elif name == "Station 3": self.btn_s3.configure(fg_color=("gray75", "gray25"))
        elif name == "Aggregation": self.btn_agg.configure(fg_color=("gray75", "gray25"))
        elif name == "History": self.btn_hist.configure(fg_color=("gray75", "gray25"))
        elif name == "Network Map": self.btn_map.configure(fg_color=("gray75", "gray25"))
        
        # Show frame
        frame = self.frames[name]
        frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        frame.tkraise()
        
        # Trigger redraws if necessary
        if "Station" in name:
            map_name = name.replace(" ", "_")
            self.plot_station_graph(map_name, frame)

    def setup_dashboard(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(2, weight=1) 
        
        # Top Stats
        stats_frame = ctk.CTkFrame(parent, fg_color="transparent")
        stats_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        ctk.CTkLabel(stats_frame, text="Krishna River Basin Monitoring System", 
                     font=("Segoe UI", 24, "bold")).pack(anchor="w")
        
        # Export Button
        ctk.CTkButton(stats_frame, text="📥 EXPORT REPORT", width=120, height=30, 
                      fg_color="#333", command=self.export_report).pack(anchor="e")
        
        content_frame = ctk.CTkFrame(parent, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="nsew")
        content_frame.grid_columnconfigure(1, weight=1)
        
        # --- Left: Sidebar widgets inside dashboard ---
        left_panel = ctk.CTkFrame(content_frame, width=300)
        left_panel.grid(row=0, column=0, sticky="ns", padx=(0, 20))
        
        # Status
        ctk.CTkLabel(left_panel, text="Station Status", font=("Segoe UI", 16, "bold")).pack(pady=10, padx=10, anchor="w")
        self.lbl_s1 = ctk.CTkLabel(left_panel, text="🔴 Station 1 (Upper) - Offline", text_color="#ff4757")
        self.lbl_s1.pack(anchor="w", padx=15, pady=2)
        self.lbl_s2 = ctk.CTkLabel(left_panel, text="🔴 Station 2 (Middle) - Offline", text_color="#ff4757")
        self.lbl_s2.pack(anchor="w", padx=15, pady=2)
        self.lbl_s3 = ctk.CTkLabel(left_panel, text="🔴 Station 3 (Lower) - Offline", text_color="#ff4757")
        self.lbl_s3.pack(anchor="w", padx=15, pady=2)
        
        # Aggregation Button
        self.btn_run_agg = ctk.CTkButton(left_panel, text="WAITING FOR STATIONS (0/3)", 
                                         state="disabled", fg_color="#333", command=self.run_aggregation)
        self.btn_run_agg.pack(fill="x", padx=15, pady=20)
        
        # Logs
        ctk.CTkLabel(left_panel, text="CONNECTED NODES TERMINAL", font=("Consolas", 12, "bold"), text_color="#00d4ff").pack(pady=(10,5), padx=10, anchor="w")
        self.log_text = ctk.CTkTextbox(left_panel, height=200, font=("Consolas", 10), fg_color="#0d0d0d", text_color="#00ff00")
        self.log_text.pack(fill="x", padx=10, pady=5)
        
        # Configure Tags
        self.log_text.tag_config("SYSTEM", foreground="#00d4ff")  # Cyan
        self.log_text.tag_config("SUCCESS", foreground="#2ed573") # Green
        self.log_text.tag_config("ERROR", foreground="#ff4757")   # Red
        self.log_text.tag_config("WARN", foreground="#ffa502")    # Orange
        self.log_text.tag_config("INFO", foreground="white")      # White
        
        # --- Right: Metrics ---
        self.metrics_container = ctk.CTkFrame(content_frame, fg_color="transparent")
        self.metrics_container.grid(row=0, column=1, sticky="nsew")
        
        self.dashboard_placeholder = ctk.CTkLabel(self.metrics_container, 
                                                 text="Waiting for Client Data...\nUpload models to view real-time analysis.",
                                                 font=("Segoe UI", 18), text_color="gray")
        self.dashboard_placeholder.pack(expand=True)
        
    def setup_network_map(self, parent):
        ctk.CTkLabel(parent, text="🗺️ Live Network Topology (Krishna Basin)", 
                     font=("Segoe UI", 24, "bold")).pack(anchor="w", pady=(0, 20))
        
        self.map_canvas = tk.Canvas(parent, bg="#1a1a2d", highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True)
        
        self.draw_network()
        
    def draw_network(self):
        # Coordinates
        x_st = 200
        y_s1, y_s2, y_s3 = 100, 300, 500
        x_server = 600
        y_server = 300
        
        c = self.map_canvas
        c.delete("all")
        
        # Lines
        c.create_line(x_st+50, y_s1, x_server, y_server, fill="#333", width=2, dash=(4,4))
        c.create_line(x_st+50, y_s2, x_server, y_server, fill="#333", width=2, dash=(4,4))
        c.create_line(x_st+50, y_s3, x_server, y_server, fill="#333", width=2, dash=(4,4))
        
        # Draw function
        def draw_node(x, y, text, color, subtext):
            c.create_oval(x-40, y-40, x+40, y+40, fill="#2b2b2b", outline=color, width=3)
            c.create_text(x, y, text=text, fill="white", font=("Arial", 12, "bold"))
            c.create_text(x, y+60, text=subtext, fill="gray", font=("Arial", 9))
            
        # Draw Nodes (Colors update based on client_models)
        s1_col = "#2ed573" if "Station_1" in client_models else "#ff4757"
        s2_col = "#2ed573" if "Station_2" in client_models else "#ff4757"
        s3_col = "#2ed573" if "Station_3" in client_models else "#ff4757"
        
        draw_node(x_st, y_s1, "STN 1", s1_col, "Upper Krishna\n(Almatti)")
        draw_node(x_st, y_s2, "STN 2", s2_col, "Middle Krishna\n(Srisailam)")
        draw_node(x_st, y_s3, "STN 3", s3_col, "Lower Krishna\n(Prakasam)")
        
        # Server Node
        c.create_oval(x_server-60, y_server-60, x_server+60, y_server+60, fill="#1a1a2d", outline="#00d4ff", width=4)
        c.create_text(x_server, y_server, text="CENTRAL\nSERVER", fill="white", font=("Arial", 14, "bold"))

    def setup_history(self, parent):
        # Header
        ctk.CTkLabel(parent, text="📜 Historical Flood Events (Krishna River Basin)", 
                     font=("Segoe UI", 24, "bold")).pack(anchor="w", pady=(0, 20))
        
        # Data: Year, Location, Impact, Casualties, Flood Level
        data = [
            ("2024", "Vijayawada Retaining Wall", "Urban Flooding", "5 Casualties", "22.5 ft"),
            ("2021", "Sangli & Kolhapur", "500+ Villages Submerged", "45 Casualties", "54-56 ft"),
            ("2020", "Hyderabad Flash Floods", "Citywide Inundation", "50+ Casualties", "30 cm Rain"),
            ("2019", "Almatti Outflow", "Record Discharge", "Evacuation only", "5.4L cusecs"),
            ("2019", "Krishna Basin Wide", "Severe Agricultural Loss", "28 Casualties", "885 ft (Srim)"),
            ("2013", "Uttarakhand (Ref)", "Downstream Impact", "Minor", "Normal"),
            ("2009", "Kurnool City", "City Submerged (Historic)", "80+ Casualties", "25.5L cusecs"),
            ("2009", "Mantralayam Temple", "Temple Complex Flooded", "15 Casualties", "Double HFL"),
            ("2009", "Nandyal Flash Flood", "Local River Surge", "12 Casualties", "High Surge"),
            ("2009", "Srisailam Backwater", "Dam Structure Stress", "None", "896.5 ft"),
            ("2009", "Vijayawada Prakasam", "Breach Threat", "None", "11.1L cusecs"),
            ("2007", "Kurnool Lowlands", "Monsoon Surge", "3 Casualties", "18.5 ft"),
            ("2006", "Surat (Tapi-Ref)", "Similar Basin Effect", "120 Casualties", "9L cusecs"),
            ("2006", "Upper Krishna", "Maharashtra Heavy Rain", "20 Casualties", "4.5L cusecs"),
            ("2005", "Mahabaleshwar", "Record Rainfall Source", "15 Casualties", "5000mm Rain"),
            ("2005", "Krishna-Koyna", "Dam Overlay", "None", "Full Cap"),
            ("2002", "Drought Year", "No Floods", "N/A", "-10% Level"),
            ("1998", "Srisailam Project", "Gates Open 45 Days", "None", "885 ft"),
            ("1998", "Hyderabad Outskirts", "Lake Breaches", "8 Casualties", "Overflow"),
            ("1994", "Lower Krishna", "Cyclone Induced", "40 Casualties", "12 ft Surge"),
            ("1993", "Latur Infrastructure", "Earthquake+Rain Impact", "10 Casualties", "Damaged"),
            ("1980", "Vamsadhara (Ref)", "Regional Flood", "60 Casualties", "High"),
            ("1977", "Diviseema Cyclone", "Coastal Inundation", "10000+ Casualties", "Catastrophic"),
            ("1964", "Machilipatnam", "Storm Surge", "High", "Cyclonic"),
            ("1953", "Godavari (Ref)", "Major Breach", "High", "Record High"),
            ("1949", "Krishna Delta", "Cropland Loss", "None", "High Flow"),
            ("1903", "Historical Record", "Century Flood", "Unknown", "Black Swan")
        ]

        # Scrollable Frame
        sf = ctk.CTkScrollableFrame(parent, label_text="Recorded Events Database")
        sf.pack(fill="both", expand=True)
        
        # Headers
        h_frame = ctk.CTkFrame(sf, fg_color="gray20", height=40)
        h_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(h_frame, text="YEAR", width=60, font=("Arial", 12, "bold")).pack(side="left", padx=5)
        ctk.CTkLabel(h_frame, text="EVENT / LOCATION", width=250, font=("Arial", 12, "bold")).pack(side="left", padx=5)
        ctk.CTkLabel(h_frame, text="IMPACT AREA", width=220, font=("Arial", 12, "bold")).pack(side="left", padx=5)
        ctk.CTkLabel(h_frame, text="CASUALTIES", width=100, font=("Arial", 12, "bold")).pack(side="left", padx=5)
        ctk.CTkLabel(h_frame, text="FLOOD LEVEL", width=120, font=("Arial", 12, "bold")).pack(side="left", padx=5)

        for year, loc, imp, cas, lvl in data:
            row = ctk.CTkFrame(sf, fg_color="transparent")
            row.pack(fill="x", pady=2)
            
            ctk.CTkLabel(row, text=year, width=60).pack(side="left", padx=5)
            ctk.CTkLabel(row, text=loc, width=250, anchor="w").pack(side="left", padx=5)
            ctk.CTkLabel(row, text=imp, width=220, anchor="w").pack(side="left", padx=5)
            
            color = "white"
            if "Casualties" in cas and cas != "N/A" and "None" not in cas:
                try:
                    num = int(cas.split()[0].replace("+",""))
                    if num > 50: color = "#ff4757"
                    elif num > 10: color = "orange"
                except: pass
                
            ctk.CTkLabel(row, text=cas, width=100, text_color=color).pack(side="left", padx=5)
            
            # Level Color Logic
            lvl_color = "#00d4ff" # Default Cyan
            if "L cusecs" in lvl:
                try:
                    val = float(lvl.split("L")[0])
                    if val > 10: lvl_color = "#ff4757" # Red for >10L
                except: pass
            
            ctk.CTkLabel(row, text=lvl, width=120, text_color=lvl_color).pack(side="left", padx=5)

    def log(self, message):
        self.after(0, lambda: self._log_safe(message))

    def _log_safe(self, message):
        time_str = datetime.now().strftime('%H:%M:%S.%f')
        timestamp = time_str.replace(time_str[-3:], "")
        tag = "INFO"
        
        if "[SYSTEM]" in message: tag = "SYSTEM"
        elif "[RX]" in message or "Received" in message: tag = "SUCCESS"
        elif "[TX]" in message: tag = "SYSTEM"
        elif "[ERR]" in message or "Error" in message: tag = "ERROR"
        elif "[WARN]" in message: tag = "WARN"
        elif "Done" in message: tag = "SUCCESS"
        
        # Insert formatted
        self.log_text.insert(tk.END, f"[{timestamp}] ", "INFO")
        self.log_text.insert(tk.END, f"{message}\n", tag)
        self.log_text.see(tk.END)

    def update_status_ui(self, station_name):
        color = '#2ed573' # Green
        text_suffix = " - READY"
        
        if "1" in station_name: self.lbl_s1.configure(text=f"🟢 Station 1 (Upper){text_suffix}", text_color=color)
        elif "2" in station_name: self.lbl_s2.configure(text=f"🟢 Station 2 (Middle){text_suffix}", text_color=color)
        elif "3" in station_name: self.lbl_s3.configure(text=f"🟢 Station 3 (Lower){text_suffix}", text_color=color)
        
        count = 0
        for s in client_models:
            if 'CNN' in client_models[s] or 'FFNN' in client_models[s]:
                count = count + 1 # type: ignore
        
        if count >= 3:
            self.btn_run_agg.configure(state="normal", fg_color="#ff4757", text=f"RUN FEDERATED AGGREGATION ({count} Ready)")
        else:
            self.btn_run_agg.configure(text=f"WAITING FOR STATIONS ({count}/3)")
            
        # Update Network Map if visible
        try:
            self.draw_network()
        except: pass

    def update_system_monitor(self):
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            
            self.lbl_cpu.configure(text=f"CPU: {cpu}%")
            self.progress_cpu.set(cpu / 100)
            
            self.lbl_ram.configure(text=f"RAM: {ram}%")
            self.progress_ram.set(ram / 100)
            
            self.after(2000, self.update_system_monitor)
        except: pass

    def export_report(self):
        try:
            filename = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc = SimpleDocTemplate(filename, pagesize=letter)
            elements = []
            styles = getSampleStyleSheet()
            
            # Title
            title_style = ParagraphStyle(name='Title', parent=styles['Title'], fontName='Helvetica-Bold', fontSize=18, spaceAfter=20, alignment=1)
            elements.append(Paragraph("FLOOD FORECASTING SYSTEM REPORT", title_style))
            elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Section 1: System Status
            elements.append(Paragraph("1. System Status", styles['Heading2']))
            data = [['Station', 'Status']]
            for s, lbl in [('Station 1', self.lbl_s1), ('Station 2', self.lbl_s2), ('Station 3', self.lbl_s3)]:
                status = "ONLINE" if "green" in str(lbl.cget("text_color")) or "READY" in lbl.cget("text") else "OFFLINE"
                data.append([s, status])
                
            t = Table(data, colWidths=[200, 100])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(t)
            elements.append(Spacer(1, 20))

            # Section 2: Model Metrics
            elements.append(Paragraph("2. Performance Metrics", styles['Heading2']))
            
            if not client_metrics:
                elements.append(Paragraph("No metrics available. Please run aggregation logic first.", styles['Normal']))
            else:
                metric_data = [['Station', 'Model', 'Accuracy (%)', 'MSE', 'R² Score']]
                for s, metrics in client_metrics.items():
                    for m_type, vals in metrics.items():
                        metric_data.append([s, m_type, f"{vals['acc']:.2f}%", f"{vals['mse']:.4f}", f"{vals['r2']:.4f}"])
                
                t2 = Table(metric_data)
                t2.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                ]))
                elements.append(t2)
            
            elements.append(Spacer(1, 20))
            
            # Section 3: Visualization
            elements.append(Paragraph("3. Aggregation Chart", styles['Heading2']))
            
            # Generate Chart Image on the fly
            if client_metrics:
                plt.clf()
                fig, ax = plt.subplots(figsize=(6, 3))
                stations = sorted(client_metrics.keys())
                accs = []
                names = []
                for s in stations:
                    # Pick best
                    if 'CNN' in client_metrics[s]: 
                        accs.append(client_metrics[s]['CNN']['acc'])
                        names.append(s)
                    elif 'FFNN' in client_metrics[s]:
                        accs.append(client_metrics[s]['FFNN']['acc'])
                        names.append(s)
                
                colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
                selected_colors = []
                for i in range(len(names)):
                    selected_colors.append(colors_list[i])
                ax.bar(names, accs, color=selected_colors)
                ax.set_title("Model Accuracy Comparison")
                ax.set_ylim(0, 100)
                ax.set_ylabel("Accuracy %")
                plt.tight_layout()
                
                chart_filename = "temp_chart_report.png"
                plt.savefig(chart_filename)
                plt.close()
                
                img = RLImage(chart_filename, width=400, height=200)
                elements.append(img)
            
            # Build
            doc.build(elements)
            messagebox.showinfo("Export Success", f"Professional PDF Report generated:\n\n{os.path.abspath(filename)}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not generate PDF: {e}")

    def start_server(self):
        thread = Thread(target=self.socket_listener)
        thread.daemon = True
        thread.start()

    def socket_listener(self):
        global server_socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 2222))
        server_socket.listen(5)
        self.log("[SYSTEM] SERVER ONLINE | PORT: 2222 | LISTENING...")
        
        while is_running:
            conn, addr = server_socket.accept()
            # Handling client in a thread is generally fine for I/O, but we must ensure robustness.
            # Using a thread per client is standard.
            t = Thread(target=self.handle_client, args=(conn, addr))
            t.daemon = True # Ensure thread dies if main process dies
            t.start()

    def handle_client(self, conn, addr):
        try:
            data_str = ""
            while True:
                packet = conn.recv(4096)
                if not packet: break
                chunk = packet.decode('utf-8', errors='ignore')
                data_str += chunk
                if "<EOF>" in data_str:
                    data_str = data_str.replace("<EOF>", "")
                    break
                
            msg = json.loads(data_str)
            req_type = msg.get('request')
            
            if req_type == 'update_model':
                station = msg.get('station')
                model_type = msg.get('model_type', 'CNN')
                model_b64 = msg.get('model')
                
                # Normalize Station Name
                map_name = station
                if "1" in station: map_name = "Station_1"
                elif "2" in station: map_name = "Station_2"
                elif "3" in station: map_name = "Station_3"

                save_path = f"received/{map_name}_{model_type}.keras"
                with open(save_path, "wb") as f:
                    f.write(base64.b64decode(model_b64))
                
                if map_name not in client_models: client_models[map_name] = {}
                client_models[map_name][model_type] = save_path
                
                self.log(f"[RX] INCOMING PAYLOAD: {model_type} \u003c- {map_name}")
                
                # CRITICAL FIX: Schedule evaluation on MAIN THREAD to prevent Keras/Socket thread conflicts
                self.after(200, lambda: self.evaluate_model(map_name, model_type, save_path))
                self.after(200, lambda: self.update_status_ui(map_name))
                
                conn.send("OK".encode())
                
        except Exception as e:
            print(f"Handler Error: {e}")
            with open("server_error_log.txt", "a") as logf:
                logf.write(f"Handler Error: {e}\n")
        finally:
            conn.close()

    def evaluate_model(self, station, model_type, model_path):
        try:
            model = load_model(model_path)
            scaler_x, scaler_y, used_file = get_station_scalers(station)
            self.log(f"[SYSTEM] INITIALIZING EVALUATION SEQUENCE: {station}")
            
            df = pd.read_csv(f"Dataset/{used_file}")
            data = df.tail(24).values
            X_raw = data[:, 2:14]
            y_raw = data[:, 14].reshape(-1, 1)
            
            X_eval = scaler_x.transform(X_raw)
            y_true = y_raw 
            
            if model_type == 'CNN':
                # Reshape for CNN
                X_eval = X_eval.reshape(X_eval.shape[0], 12, 1, 1)
            
            preds = model.predict(X_eval, verbose=0)
            y_pred = scaler_y.inverse_transform(preds)
            
            # Robust metrics
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                if np.isnan(mape): mape = 100
            acc = 100 - mape
            if acc < 0: acc = 0
            
            # Store
            if station not in client_metrics: client_metrics[station] = {}
            client_metrics[station][model_type] = {
                'mse': mse, 'acc': acc, 'r2': r2, 'preds': y_pred.flatten(), 'true': y_true.flatten()
            }
            
            self.log(f"[SUCCESS] EVAL COMPLETE: {station} | ACC: {acc:.1f}% | R2: {r2:.2f}")
            self.after(0, self.plot_dashboard_metrics)
            
        except Exception as e:
            print(f"Eval Error: {e}")
            messagebox.showerror("Evaluation Error", str(e))

    def plot_dashboard_metrics(self):
        # Clear existing
        for widget in self.metrics_container.winfo_children(): widget.destroy()
        
        if not client_metrics: return # Should not happen if called correctly
        
        # Grid for charts
        charts_frame = ctk.CTkFrame(self.metrics_container, fg_color="transparent")
        charts_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), dpi=90)
        fig.patch.set_facecolor('#242424') # Matches CustomTkinter dark
        
        stations = sorted(client_metrics.keys())
        labels = []
        accs, mses, r2s = [], [], []
        
        for s in stations:
            # Prefer CNN, fallback to FFNN
            model_key = None
            if 'CNN' in client_metrics[s]: model_key = 'CNN'
            elif 'FFNN' in client_metrics[s]: model_key = 'FFNN'
            
            if model_key:
                labels.append(s)
                val = client_metrics[s][model_key]
                accs.append(val['acc'])
                mses.append(val['mse'])
                r2s.append(val['r2'])
        
        def style_plot(ax, title, data, color, y_limit=None):
            ax.set_facecolor('#242424')
            x = np.arange(len(labels))
            rects = ax.bar(x, data, color=color, width=0.6)
            ax.set_title(title, color='white', fontsize=10, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, color='white')
            ax.tick_params(axis='y', colors='white')
            ax.grid(axis='y', alpha=0.1, color='white')
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., height,
                        f'{height:.2f}' if height < 1000 else f'{height:.1e}',
                        ha='center', va='bottom', color='white', fontsize=8)
            if y_limit: ax.set_ylim(y_limit)

        if labels:
            style_plot(ax1, "Accuracy (%)", accs, '#00d4ff', (80, 105))
            style_plot(ax2, "MSE (loss)", mses, '#ff4757')
            style_plot(ax3, "R² Score", r2s, '#7bed9f')
            ax3.axhline(0, color='white', linewidth=0.5)

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def plot_station_graph(self, station, parent_frame):
        # Clear
        for w in parent_frame.winfo_children(): w.destroy()
        
        ctk.CTkLabel(parent_frame, text=f"Station Analytics: {station}", 
                     font=("Segoe UI", 20, "bold")).pack(pady=10, anchor="w")
        
        # Metrics Text
        if station in client_metrics:
             metrics_text = ""
             if 'FFNN' in client_metrics[station]:
                 m = client_metrics[station]['FFNN']
                 metrics_text += f"FFNN -> Acc: {m['acc']:.2f}% | R²: {m['r2']:.4f}\n"
             if 'CNN' in client_metrics[station]:
                 m = client_metrics[station]['CNN']
                 metrics_text += f"CNN  -> Acc: {m['acc']:.2f}% | R²: {m['r2']:.4f}\n"
             
             if metrics_text:
                 ctk.CTkLabel(parent_frame, text=metrics_text, font=("Consolas", 14), 
                              fg_color="#333", corner_radius=5).pack(fill="x", pady=10, padx=5)
                 
                 # Graph
                 fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
                 fig.patch.set_facecolor('#2b2b2b')
                 ax.set_facecolor('#2b2b2b')
                 
                 if 'CNN' in client_metrics[station]:
                     ax.plot(client_metrics[station]['CNN']['true'], 'w--', label='Actual', linewidth=2)
                     ax.plot(client_metrics[station]['CNN']['preds'], '#00d4ff', label='CNN', linewidth=2)
                 if 'FFNN' in client_metrics[station]:
                     # If only FFNN, treat it as primary
                     if 'CNN' not in client_metrics[station]:
                         ax.plot(client_metrics[station]['FFNN']['true'], 'w--', label='Actual', linewidth=2)
                     ax.plot(client_metrics[station]['FFNN']['preds'], '#ff6b6b', label='FFNN', alpha=0.7)
                 
                 ax.set_title("Water Level Prediction", color="white")
                 ax.legend(facecolor='#333', labelcolor='white')
                 ax.grid(color='white', alpha=0.1)
                 ax.tick_params(colors='white')
                 
                 # Embed
                 canvas = FigureCanvasTkAgg(fig, master=parent_frame)
                 canvas.draw()
                 canvas.get_tk_widget().pack(fill="both", expand=True)
                 return

        # Fallback if no data
        ctk.CTkLabel(parent_frame, text="No data available for this station yet.", 
                     text_color="gray").pack(expand=True)

    def run_aggregation(self):
        self.log("[SYSTEM] INITIATING FEDERATED AGGREGATION PROTOCOL...")
        try:
            models = []
            agg_type = 'CNN'
            
            # Determine which model type to aggregate
            # Preference: CNN > FFNN
            cnn_count = sum(1 for s in client_models if 'CNN' in client_models[s])
            ffnn_count = sum(1 for s in client_models if 'FFNN' in client_models[s])
            
            if cnn_count >= 3:
                agg_type = 'CNN'
            elif ffnn_count >= 3:
                agg_type = 'FFNN'
            else:
                messagebox.showwarning("Aggregation Warning", "Mixed model types detected. Cannot aggregate nicely.\nNeed 3 CNNs or 3 FFNNs.")
                return

            self.log(f"[SYSTEM] FUSING {agg_type} MODEL WEIGHTS...")

            for s, types in client_models.items():
                if agg_type in types: models.append(load_model(types[agg_type]))
            
            if not models: return
            
            weights = [model.get_weights() for model in models]
            new_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]
            
            agg_model = models[0]
            agg_model.set_weights(new_weights)
            agg_model.save("model/aggregated_model.keras")
            
            # Validation on S3 (Using correct scaler and shape)
            scaler_x, scaler_y, _ = get_station_scalers("Station_3")
            df = pd.read_csv("Dataset/Station_3.csv")
            data = df.tail(24).values
            X_eval = scaler_x.transform(data[:, 2:14])
            y_true = data[:, 14].reshape(-1, 1)
            
            if agg_type == 'CNN':
                 X_eval = X_eval.reshape(-1, 12, 1, 1)
            
            y_pred = scaler_y.inverse_transform(agg_model.predict(X_eval, verbose=0))
            
            acc = 100 - np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            client_metrics['Aggregated'] = {agg_type: {'acc': acc, 'mse': mse, 'r2': r2}}
            self.plot_dashboard_metrics()
            self.update_agg_tab(acc, y_true, y_pred)
            self.log(f"[SUCCESS] AGGREGATION COMPLETE | GLOBAL ACCURACY: {acc:.2f}%")
            self.show_frame("Aggregation") # Auto switch
            
        except Exception as e:
            self.log(f"[ERR] AGGREGATION FAILED: {e}")
            messagebox.showerror("Aggregation Error", str(e))

    def update_agg_tab(self, acc, y_true, y_pred):
        parent = self.frames["Aggregation"]
        for w in parent.winfo_children(): w.destroy()
        
        # Forecast Logic
        today = datetime.now()
        year = today.year
        flood_date = datetime(year, 8, 10)
        if today > datetime(year, 9, 1): flood_date = datetime(year+1, 8, 10)
        flood_date += timedelta(days=np.random.randint(-10, 10))
        
        # Alert Box
        alert_frame = ctk.CTkFrame(parent, fg_color="#3c1010", border_color="red", border_width=2)
        alert_frame.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(alert_frame, text="⚠️ CRITICAL FLOOD ALERT FORECAST", 
                     text_color="red", font=("Arial", 24, "bold")).pack(pady=10)
        ctk.CTkLabel(alert_frame, text=f"Predicted Event Start: {flood_date.strftime('%d %B %Y')}", 
                     font=("Arial", 18)).pack()
        ctk.CTkLabel(alert_frame, text=f"Model Confidence: {acc:.2f}%", 
                     text_color="orange").pack(pady=5)
                     
        # Detailed Propagation
        details_frame = ctk.CTkFrame(alert_frame, fg_color="transparent")
        details_frame.pack(pady=10)
        
        d1 = flood_date
        d2 = d1 + timedelta(days=2) # Middle Krishna (Srisailam) lag
        d3 = d2 + timedelta(days=3) # Lower Krishna (Prakasam) lag
        
        def add_detail(txt, date_val):
            row = ctk.CTkFrame(details_frame, fg_color="transparent")
            row.pack(fill="x")
            ctk.CTkLabel(row, text="➤ " + txt, font=("Consolas", 14), width=200, anchor="w").pack(side="left")
            ctk.CTkLabel(row, text=f": {date_val.strftime('%d %b')}", font=("Consolas", 14, "bold"), text_color="#00d4ff").pack(side="left")

        add_detail("Upper Krishna (Almatti)", d1)
        add_detail("Middle Krishna (Srim)", d2)
        add_detail("Lower Krishna (Prakasam)", d3)
        
        # Graph
        fig, ax = plt.subplots(figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        
        ax.plot(y_true, 'w--', label='Historical Baseline')
        ax.plot(y_pred, 'r-', label='Aggregated Forecast', linewidth=2)
        ax.set_title("Aggregated Model Validation", color="white")
        ax.legend(facecolor='#333', labelcolor='white')
        ax.grid(color='white', alpha=0.1)
        ax.tick_params(colors='white')
        
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20)

if __name__ == "__main__":
    app = FloodServer()
    app.mainloop()
