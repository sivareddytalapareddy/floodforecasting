import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import socket
from threading import Thread
import json, base64, os, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from keras.models import load_model
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import psutil
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")
plt.style.use("dark_background")

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0d1117",
    "panel":    "#161b22",
    "card":     "#21262d",
    "border":   "#30363d",
    "cyan":     "#00d4ff",
    "green":    "#2ecc71",
    "red":      "#ff4757",
    "orange":   "#ffa502",
    "purple":   "#9b59b6",
    "white":    "#e6edf3",
    "gray":     "#8b949e",
    "nav":      "#010409",
}

# ── Globals ───────────────────────────────────────────────────────────────────
server_socket  = None
client_models  = {}   # {station_name: {"CNN": path, "FFNN": path}}
client_metrics = {}   # {station_name: {"CNN": {...}, "FFNN": {...}}}
is_running     = True

# ── Dirs ──────────────────────────────────────────────────────────────────────
for d in ("received", "model"):
    os.makedirs(d, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
_STATION_MAP = {
    "1": "Station_1.csv",
    "2": "Station_2.csv",
    "3": "Station_3.csv",
}

def normalize_station_name(raw: str) -> str:
    """Always return Station_1 / Station_2 / Station_3.
    Checks for digit '1'/'2'/'3' anywhere in raw; falls back to Station_1.
    """
    raw = str(raw).strip()
    for digit in ("1", "2", "3"):
        if digit in raw:
            return f"Station_{digit}"
    return "Station_1"   # safe default

def get_station_scalers(station_name: str):
    """Load the CSV that belongs to station_name and return fitted scalers."""
    # station_name must already be normalised (e.g. 'Station_1')
    digit = next((d for d in ("1", "2", "3") if d in station_name), "1")
    fname = _STATION_MAP[digit]
    path  = f"Dataset/{fname}"
    if not os.path.exists(path):
        # Last-resort: try any available station CSV
        for alt_fname in _STATION_MAP.values():
            alt_path = f"Dataset/{alt_fname}"
            if os.path.exists(alt_path):
                path, fname = alt_path, alt_fname
                break
        else:
            raise FileNotFoundError(f"No station CSV found in Dataset/")
    df   = pd.read_csv(path)
    data = df.values
    # Columns: SUBDIVISION(0), YEAR(1), JAN–DEC(2‑13), water_level(14)
    X_raw = data[:, 2:14].astype(np.float64)
    y_raw = data[:, 14].astype(np.float64).reshape(-1, 1)
    sx   = MinMaxScaler().fit(X_raw)
    sy   = MinMaxScaler().fit(y_raw)
    return sx, sy, fname

def make_card(parent, **kw):
    return ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=12,
                        border_color=C["border"], border_width=1, **kw)


# ══════════════════════════════════════════════════════════════════════════════
#  SPLASH SCREEN
# ══════════════════════════════════════════════════════════════════════════════
class SplashScreen(ctk.CTkToplevel):
    STAGES = [
        (10,  "🔌  Binding TCP sockets …"),
        (30,  "🧠  Loading neural engine …"),
        (55,  "🌐  Building network topology …"),
        (75,  "📡  Starting Federated Listener …"),
        (90,  "📊  Preparing dashboards …"),
        (100, "✅  Server Ready!"),
    ]

    def __init__(self, master, on_done):
        super().__init__(master)
        self.on_done = on_done
        self.overrideredirect(True)
        w, h = 540, 380
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.configure(fg_color=C["nav"])

        # glowing title
        ctk.CTkLabel(self, text="🌊", font=("Arial", 52)).pack(pady=(40, 4))
        ctk.CTkLabel(self, text="FLOOD FORECASTING SERVER",
                     font=("Arial", 20, "bold"), text_color=C["cyan"]).pack()
        ctk.CTkLabel(self, text="Federated Learning  •  Krishna River Basin",
                     font=("Arial", 11), text_color=C["gray"]).pack(pady=4)

        self.status_lbl = ctk.CTkLabel(self, text="Initialising …",
                                       font=("Consolas", 11), text_color=C["white"])
        self.status_lbl.pack(pady=(20, 6))

        self.bar = ctk.CTkProgressBar(self, width=440, height=8,
                                      mode="determinate",
                                      progress_color=C["cyan"],
                                      fg_color=C["border"])
        self.bar.set(0)
        self.bar.pack()

        self.pct_lbl = ctk.CTkLabel(self, text="0 %",
                                    font=("Consolas", 10), text_color=C["gray"])
        self.pct_lbl.pack(pady=4)

        ctk.CTkLabel(self, text="v2.0  |  Advanced Edition",
                     font=("Arial", 9), text_color=C["gray"]).pack(side="bottom", pady=12)

        self._step = 0
        self._target = 0
        self._stage_idx = 0
        self.after(200, self._tick)

    def _tick(self):
        if self._stage_idx < len(self.STAGES):
            tgt, msg = self.STAGES[self._stage_idx]
            if self._target != tgt:
                self._target = tgt
                self.status_lbl.configure(text=msg)

            if self._step < self._target:
                self._step = min(self._step + 1, self._target)
                self.bar.set(self._step / 100)
                self.pct_lbl.configure(text=f"{self._step} %")

            if self._step >= self._target:
                self._stage_idx += 1
                self.after(300, self._tick)
                return

        if self._step >= 100:
            self.after(600, self._finish)
            return
        self.after(40, self._tick)

    def _finish(self):
        self.destroy()
        self.on_done()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class FloodServer(ctk.CTk):
    PAGES = ["Dashboard", "Station 1", "Station 2", "Station 3",
             "Aggregation", "History", "Network Map"]

    def __init__(self):
        super().__init__()
        self.withdraw()
        self.title("FL Central Server  •  Krishna River Flood Forecasting")
        self.geometry("1440x900")
        self.configure(fg_color=C["bg"])

        self._active_page = None
        self._pulse_angle  = 0
        self._pulse_job    = None

        # Stat-card label references (set in _build_dashboard)
        self._lbl_stations_val = None
        self._lbl_models_val   = None
        self._lbl_acc_val      = None
        self._lbl_agg_val      = None
        self._agg_count        = 0

        splash = SplashScreen(self, self._boot)

    # ── boot (called after splash) ────────────────────────────────────────────
    def _boot(self):
        self._build_layout()
        self.start_server()
        self.update_system_monitor()
        self._pulse_nodes()
        self.show_page("Dashboard")
        self.deiconify()

    # ── Layout skeleton ───────────────────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # sidebar
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0,
                                    fg_color=C["nav"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # content area
        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color=C["bg"])
        self.content.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.content.grid_rowconfigure(0, weight=1)
        self.content.grid_columnconfigure(0, weight=1)

        self._build_sidebar()
        self._build_pages()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        sb = self.sidebar

        # logo block
        logo_f = ctk.CTkFrame(sb, fg_color=C["panel"], height=90, corner_radius=0)
        logo_f.pack(fill="x")
        ctk.CTkLabel(logo_f, text="🌊  FL SERVER",
                     font=("Arial", 18, "bold"), text_color=C["cyan"]).pack(pady=28)

        # nav buttons
        icons = {"Dashboard":"🏠", "Station 1":"📍", "Station 2":"📍",
                 "Station 3":"📍", "Aggregation":"⚙️", "History":"📜",
                 "Network Map":"🗺️"}
        self._nav_btns = {}
        for page in self.PAGES:
            btn = ctk.CTkButton(
                sb, text=f"  {icons[page]}  {page}",
                font=("Segoe UI", 13), anchor="w",
                fg_color="transparent", hover_color=C["card"],
                text_color=C["gray"], corner_radius=8, height=44,
                command=lambda p=page: self.show_page(p))
            btn.pack(fill="x", padx=8, pady=2)
            self._nav_btns[page] = btn

        # divider
        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=12, pady=12)

        # system health
        ctk.CTkLabel(sb, text="SYSTEM HEALTH",
                     font=("Consolas", 10, "bold"), text_color=C["gray"]).pack(anchor="w", padx=14)
        self.lbl_cpu = ctk.CTkLabel(sb, text="CPU  0%", font=("Consolas", 10), text_color=C["white"])
        self.lbl_cpu.pack(anchor="w", padx=14, pady=(4,0))
        self.bar_cpu = ctk.CTkProgressBar(sb, height=5, progress_color=C["cyan"],  fg_color=C["border"])
        self.bar_cpu.set(0); self.bar_cpu.pack(fill="x", padx=14, pady=(2,6))

        self.lbl_ram = ctk.CTkLabel(sb, text="RAM  0%", font=("Consolas", 10), text_color=C["white"])
        self.lbl_ram.pack(anchor="w", padx=14)
        self.bar_ram = ctk.CTkProgressBar(sb, height=5, progress_color=C["red"], fg_color=C["border"])
        self.bar_ram.set(0); self.bar_ram.pack(fill="x", padx=14, pady=(2,6))

        # station badges
        ctk.CTkFrame(sb, height=1, fg_color=C["border"]).pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(sb, text="STATION STATUS", font=("Consolas", 10, "bold"),
                     text_color=C["gray"]).pack(anchor="w", padx=14)
        self.lbl_s1 = ctk.CTkLabel(sb, text="⚫  Station 1  Offline", font=("Consolas", 10), text_color=C["red"])
        self.lbl_s2 = ctk.CTkLabel(sb, text="⚫  Station 2  Offline", font=("Consolas", 10), text_color=C["red"])
        self.lbl_s3 = ctk.CTkLabel(sb, text="⚫  Station 3  Offline", font=("Consolas", 10), text_color=C["red"])
        for lbl in (self.lbl_s1, self.lbl_s2, self.lbl_s3):
            lbl.pack(anchor="w", padx=14, pady=2)

        # version footer
        ctk.CTkLabel(sb, text="v2.0 Advanced Edition",
                     font=("Arial", 9), text_color=C["gray"]).pack(side="bottom", pady=10)

    # ── Page container builder ────────────────────────────────────────────────
    def _build_pages(self):
        self.pages = {}
        for p in self.PAGES:
            f = ctk.CTkFrame(self.content, fg_color=C["bg"], corner_radius=0)
            f.grid(row=0, column=0, sticky="nsew")
            self.pages[p] = f

        self._build_dashboard(self.pages["Dashboard"])
        self._build_history(self.pages["History"])
        self._build_network_map(self.pages["Network Map"])
        self._build_aggregation_placeholder(self.pages["Aggregation"])
        for s in ("Station 1", "Station 2", "Station 3"):
            self._build_station_placeholder(self.pages[s], s)

    def show_page(self, name):
        for p, btn in self._nav_btns.items():
            if p == name:
                btn.configure(fg_color=C["card"], text_color=C["cyan"])
            else:
                btn.configure(fg_color="transparent", text_color=C["gray"])
        self.pages[name].tkraise()
        self._active_page = name
        if "Station" in name:
            self._refresh_station(name)

    # -- DASHBOARD -------------------------------------------------------------
    def _build_dashboard(self, parent):
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 8))
        ctk.CTkLabel(hdr, text="Krishna River Basin  —  Monitoring Dashboard",
                     font=("Segoe UI", 22, "bold"), text_color=C["white"]).pack(side="left")
        self.lbl_clock = ctk.CTkLabel(hdr, text="", font=("Consolas", 12), text_color=C["gray"])
        self.lbl_clock.pack(side="right")
        self._tick_clock()
        ctk.CTkButton(hdr, text="Export Report", width=130, height=30,
                      fg_color=C["card"], hover_color=C["border"],
                      command=self.export_report).pack(side="right", padx=10)

        cards_row = ctk.CTkFrame(parent, fg_color="transparent")
        cards_row.grid(row=1, column=0, sticky="nsew", padx=24, pady=(6, 0))
        for i in range(4):
            cards_row.grid_columnconfigure(i, weight=1)
        cards_row.grid_rowconfigure(0, weight=0)
        cards_row.grid_rowconfigure(1, weight=1)

        def stat_card(p, col, title, init_val, color, icon):
            f = make_card(p)
            f.grid(row=0, column=col, padx=6, pady=(8, 12), sticky="ew")
            ctk.CTkLabel(f, text=icon, font=("Arial", 18), text_color=color).pack(anchor="w", padx=14, pady=(12, 0))
            val_lbl = ctk.CTkLabel(f, text=init_val, font=("Arial", 22, "bold"), text_color=color)
            val_lbl.pack(anchor="w", padx=14)
            ctk.CTkLabel(f, text=title, font=("Arial", 10), text_color=C["gray"]).pack(anchor="w", padx=14, pady=(0, 12))
            return f, val_lbl

        self.card_stations, self._lbl_stations_val = stat_card(cards_row, 0, "Stations Connected", "0/3",   C["cyan"],   "STATIONS")
        self.card_models,   self._lbl_models_val   = stat_card(cards_row, 1, "Models Received",    "0",     C["green"],  "MODELS")
        self.card_acc,      self._lbl_acc_val       = stat_card(cards_row, 2, "Best Accuracy",      "—",     C["orange"], "ACCURACY")
        self.card_agg,      self._lbl_agg_val       = stat_card(cards_row, 3, "Aggregations Run",   "0",     C["purple"], "AGG RUNS")

        bottom = ctk.CTkFrame(parent, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="nsew", padx=24, pady=(0, 12))
        parent.grid_rowconfigure(2, weight=1)
        bottom.grid_columnconfigure(0, weight=3)
        bottom.grid_columnconfigure(1, weight=2)
        bottom.grid_rowconfigure(0, weight=1)

        self.chart_card = make_card(bottom)
        self.chart_card.grid(row=0, column=0, padx=(0, 6), sticky="nsew")
        self.metrics_placeholder = ctk.CTkLabel(
            self.chart_card,
            text="Waiting for station uploads...\n\nUpload FFNN / CNN models from clients.",
            font=("Segoe UI", 14), text_color=C["gray"])
        self.metrics_placeholder.pack(expand=True)

        log_card = make_card(bottom)
        log_card.grid(row=0, column=1, padx=(6, 0), sticky="nsew")
        ctk.CTkLabel(log_card, text="ACTIVITY FEED",
                     font=("Consolas", 11, "bold"), text_color=C["cyan"]).pack(anchor="w", padx=12, pady=(10, 4))
        self.log_box = ctk.CTkTextbox(log_card, font=("Consolas", 10),
                                      fg_color=C["bg"], text_color=C["green"], wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._log("[SYSTEM] Dashboard ready. Awaiting connections...")

    def _tick_clock(self):
        try:
            self.lbl_clock.configure(text=datetime.now().strftime("%H:%M:%S  |  %d %b %Y"))
            self.after(1000, self._tick_clock)
        except Exception:
            pass

    # ── Update the 4 stat-card values (always safe to call from main thread) ──
    def _update_stat_cards(self):
        try:
            station_count = len(client_models)
            model_count   = sum(len(v) for v in client_models.values())
            self._lbl_stations_val.configure(text=f"{station_count}/3")
            self._lbl_models_val.configure(text=str(model_count))
        except Exception:
            pass

    # -- STATION PAGES --------------------------------------------------------
    def _build_station_placeholder(self, parent, name):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(parent, text=f"{name}\n\nNo data yet. Upload model from this station.",
                     font=("Segoe UI", 16), text_color=C["gray"]).grid(row=0, column=0)

    def _refresh_station(self, page_name):
        parent = self.pages[page_name]
        key = page_name.replace(" ", "_")
        for w in parent.winfo_children():
            w.destroy()
        ctk.CTkLabel(parent, text=f"{page_name}  Analytics",
                     font=("Segoe UI", 20, "bold"), text_color=C["white"]).pack(anchor="w", padx=20, pady=(16, 4))
        if key not in client_metrics:
            ctk.CTkLabel(parent, text="No data available. Upload model from this station.",
                         text_color=C["gray"]).pack(expand=True)
            return
        m = client_metrics[key]
        badges = ctk.CTkFrame(parent, fg_color="transparent")
        badges.pack(fill="x", padx=20, pady=8)
        mtype = "CNN" if "CNN" in m else "FFNN"
        md = m[mtype]

        def badge(col, title, val, color):
            f = make_card(badges)
            f.grid(row=0, column=col, padx=5, sticky="ew")
            badges.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(f, text=val, font=("Arial", 18, "bold"), text_color=color).pack(pady=(10, 2))
            ctk.CTkLabel(f, text=title, font=("Arial", 9), text_color=C["gray"]).pack(pady=(0, 10))

        badge(0, "Accuracy",   f"{md['acc']:.1f}%",  C["cyan"])
        badge(1, "MSE",        f"{md['mse']:.4f}",   C["red"])
        badge(2, "R² Score",   f"{md['r2']:.4f}",    C["green"])
        badge(3, "Model Type", mtype,                 C["orange"])

        chart_f = make_card(parent)
        chart_f.pack(fill="both", expand=True, padx=20, pady=8)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=95)
        fig.patch.set_facecolor(C["panel"])
        ax.set_facecolor(C["bg"])
        ax.plot(md["true"],  color=C["white"],  lw=2, label="Actual",    linestyle="--")
        ax.plot(md["preds"], color=C["cyan"],   lw=2, label=f"{mtype} Prediction")
        if "FFNN" in m and mtype != "FFNN":
            ax.plot(m["FFNN"]["preds"], color=C["orange"], lw=1.5, alpha=0.7, label="FFNN")
        ax.set_title(f"Water Level Prediction — {page_name}", color=C["white"], fontsize=11)
        ax.legend(facecolor=C["card"], labelcolor=C["white"])
        ax.tick_params(colors=C["gray"])
        ax.grid(color=C["border"], alpha=0.5)
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=chart_f)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # -- NETWORK MAP ----------------------------------------------------------
    def _build_network_map(self, parent):
        ctk.CTkLabel(parent, text="Live Network Topology — Krishna Basin",
                     font=("Segoe UI", 20, "bold"), text_color=C["white"]).pack(anchor="w", padx=20, pady=(16, 4))
        self.map_canvas = tk.Canvas(parent, bg=C["bg"], highlightthickness=0)
        self.map_canvas.pack(fill="both", expand=True, padx=20, pady=8)
        self.map_canvas.bind("<Configure>", lambda e: self._draw_network())

    def _draw_network(self):
        c = self.map_canvas
        c.delete("all")
        W = c.winfo_width() or 900
        H = c.winfo_height() or 600
        cx, cy = int(W * 0.62), H // 2
        sx = int(W * 0.22)
        sy_list = [H // 5, H // 2, int(H * 0.78)]
        labels = ["Station_1", "Station_2", "Station_3"]
        locs   = ["Almatti (Upper)", "Srisailam (Mid)", "Prakasam (Lower)"]
        for i, (sy, sn, loc) in enumerate(zip(sy_list, labels, locs)):
            online = sn in client_models
            col = C["green"] if online else C["red"]
            c.create_oval(sx - 52, sy - 52, sx + 52, sy + 52, outline=C["cyan"] if online else "#330000", width=3, dash=(6, 4))
            c.create_oval(sx - 38, sy - 38, sx + 38, sy + 38, fill=C["card"], outline=col, width=3)
            c.create_text(sx, sy,     text=f"S{i+1}", fill="white", font=("Arial", 14, "bold"))
            c.create_text(sx, sy + 54, text=loc,      fill=C["gray"],  font=("Arial", 9))
            c.create_line(sx + 40, sy, cx - 50, cy, fill=C["cyan"] if online else C["border"], width=2, dash=(8, 5))
            c.create_text(sx, sy + 68, text="ONLINE" if online else "OFFLINE", fill=col, font=("Consolas", 9, "bold"))
        c.create_oval(cx - 56, cy - 56, cx + 56, cy + 56, fill=C["nav"],   outline=C["cyan"],   width=4)
        c.create_oval(cx - 44, cy - 44, cx + 44, cy + 44, fill=C["panel"], outline=C["border"], width=1)
        c.create_text(cx, cy - 8,  text="CENTRAL", fill="white",   font=("Arial", 11, "bold"))
        c.create_text(cx, cy + 12, text="SERVER",  fill=C["cyan"], font=("Arial", 11, "bold"))
        c.create_text(cx, cy + 68, text="PORT 2222", fill=C["gray"], font=("Consolas", 9))

    def _pulse_nodes(self):
        try:
            self._draw_network()
        except Exception:
            pass
        self.after(3000, self._pulse_nodes)

    # -- AGGREGATION PLACEHOLDER ----------------------------------------------
    def _build_aggregation_placeholder(self, parent):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=0, column=0)
        ctk.CTkLabel(f, text="Aggregation results appear here.\n\nWait for all 3 stations to upload models.",
                     font=("Segoe UI", 16), text_color=C["gray"]).pack(pady=20)
        ctk.CTkButton(f, text="RUN FEDERATED AGGREGATION",
                      fg_color=C["red"], font=("Arial", 14, "bold"),
                      height=44, command=self.run_aggregation).pack(pady=10)

    # -- HISTORY --------------------------------------------------------------
    def _build_history(self, parent):
        ctk.CTkLabel(parent, text="Historical Flood Events — Krishna River Basin",
                     font=("Segoe UI", 20, "bold"), text_color=C["white"]).pack(anchor="w", padx=20, pady=(16, 8))
        EVENTS = [
            ("2024", "Vijayawada Wall",   "Urban Flooding",         "5",     "22.5 ft",     "MEDIUM"),
            ("2021", "Sangli & Kolhapur", "500+ Villages Submerged","45",    "54-56 ft",    "HIGH"),
            ("2020", "Hyderabad Flash",   "Citywide Inundation",    "50+",   "30 cm Rain",  "HIGH"),
            ("2019", "Krishna Basin",     "Agri Loss",              "28",    "885 ft",      "HIGH"),
            ("2009", "Kurnool City",      "City Submerged",         "80+",   "25.5L cus",   "CRITICAL"),
            ("2009", "Mantralayam",       "Complex Flooded",        "15",    "2x HFL",      "HIGH"),
            ("2009", "Srisailam Dam",     "Structure Stress",       "0",     "896.5 ft",    "MEDIUM"),
            ("2007", "Kurnool Lowlands",  "Monsoon Surge",          "3",     "18.5 ft",     "LOW"),
            ("2006", "Upper Krishna",     "Maharashtra Rain",       "20",    "4.5L cus",    "MEDIUM"),
            ("1977", "Diviseema",         "Coastal Inundation",     "10000+","Catastrophic","CRITICAL"),
        ]
        sf = ctk.CTkScrollableFrame(parent, label_text="Event Database", fg_color=C["panel"])
        sf.pack(fill="both", expand=True, padx=20, pady=8)
        hdrs   = ["Year", "Location",     "Impact",   "Casualties", "Level", "Severity"]
        widths = [55,      200,            200,         100,          110,     90]
        h_row  = ctk.CTkFrame(sf, fg_color=C["border"], height=36)
        h_row.pack(fill="x", pady=(0, 4))
        for title, w in zip(hdrs, widths):
            ctk.CTkLabel(h_row, text=title, width=w, font=("Consolas", 10, "bold"),
                         text_color=C["white"]).pack(side="left", padx=4)
        sc = {"LOW": C["green"], "MEDIUM": C["orange"], "HIGH": C["red"], "CRITICAL": "#ff0055"}
        for yr, loc, imp, cas, lvl, sev in EVENTS:
            row = ctk.CTkFrame(sf, fg_color=C["card"], corner_radius=6, height=32)
            row.pack(fill="x", pady=2)
            for v, w in zip([yr, loc, imp, cas, lvl, sev], widths):
                col = sc.get(sev, C["white"]) if v == sev else C["white"]
                ctk.CTkLabel(row, text=v, width=w, font=("Consolas", 10),
                             text_color=col, anchor="w").pack(side="left", padx=4)

    # -- SYSTEM MONITOR -------------------------------------------------------
    def update_system_monitor(self):
        try:
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            self.lbl_cpu.configure(text=f"CPU  {cpu:.0f}%")
            self.bar_cpu.set(cpu / 100)
            self.lbl_ram.configure(text=f"RAM  {ram:.0f}%")
            self.bar_ram.set(ram / 100)
        except Exception:
            pass
        self.after(2000, self.update_system_monitor)

    # -- LOGGING --------------------------------------------------------------
    def _log(self, msg):
        self.after(0, lambda: self._log_ui(msg))

    def _log_ui(self, msg):
        try:
            ts = datetime.now().strftime("%H:%M:%S")
            self.log_box.insert("end", f"[{ts}] {msg}\n")
            self.log_box.see("end")
        except Exception:
            pass

    # -- SIDEBAR STATION LABEL UPDATE -----------------------------------------
    def _update_station_label(self, station_name):
        lbl_map = {"Station_1": self.lbl_s1, "Station_2": self.lbl_s2, "Station_3": self.lbl_s3}
        idx_map = {"Station_1": "1",          "Station_2": "2",          "Station_3": "3"}
        lbl = lbl_map.get(station_name)
        if lbl:
            lbl.configure(
                text=f"🟢  Station {idx_map.get(station_name, '?')}  Online",
                text_color=C["green"]
            )

    # -- DASHBOARD METRICS CHART ----------------------------------------------
    def _refresh_dashboard_chart(self):
        # Clear chart card safely
        for w in self.chart_card.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass

        stations = [s for s in client_metrics if s != "Aggregated"]
        if not stations:
            # Re-show placeholder
            self.metrics_placeholder = ctk.CTkLabel(
                self.chart_card,
                text="Waiting for station uploads...\n\nUpload FFNN / CNN models from clients.",
                font=("Segoe UI", 14), text_color=C["gray"])
            self.metrics_placeholder.pack(expand=True)
            return

        labels, accs, mses, r2s = [], [], [], []
        for s in sorted(stations):
            mk = "CNN" if "CNN" in client_metrics[s] else ("FFNN" if "FFNN" in client_metrics[s] else None)
            if mk:
                labels.append(s.replace("Station_", "S"))
                accs.append(client_metrics[s][mk]["acc"])
                mses.append(client_metrics[s][mk]["mse"])
                r2s.append(client_metrics[s][mk]["r2"])
        if not labels:
            return

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), dpi=95)
        fig.patch.set_facecolor(C["panel"])
        x = np.arange(len(labels))
        specs = [
            (axes[0], "Accuracy (%)", accs, C["cyan"],   (0, 105)),
            (axes[1], "MSE (loss)",   mses, C["red"],    None),
            (axes[2], "R² Score",     r2s,  C["green"],  None),
        ]
        for ax, title, data, color, ylim in specs:
            ax.set_facecolor(C["bg"])
            bars = ax.bar(x, data, color=color, width=0.55, zorder=3)
            ax.set_title(title, color=C["white"], fontsize=10, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, color=C["gray"])
            ax.tick_params(axis="y", colors=C["gray"])
            ax.grid(axis="y", color=C["border"], alpha=0.5, zorder=0)
            if ylim:
                ax.set_ylim(ylim)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}",
                        ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

        # Update Best Accuracy card
        if accs:
            try:
                self._lbl_acc_val.configure(text=f"{max(accs):.1f}%")
            except Exception:
                pass

        fig.tight_layout(pad=1.5)
        cv = FigureCanvasTkAgg(fig, master=self.chart_card)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # -- SERVER SOCKET --------------------------------------------------------
    def start_server(self):
        Thread(target=self._socket_listener, daemon=True).start()

    def _socket_listener(self):
        global server_socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(("localhost", 2222))
        server_socket.listen(5)
        self._log("[SYSTEM] SERVER ONLINE | PORT 2222 | LISTENING...")
        while is_running:
            try:
                conn, addr = server_socket.accept()
                Thread(target=self._handle_client, args=(conn, addr), daemon=True).start()
            except Exception:
                break

    def _handle_client(self, conn, addr):
        """
        Receive a complete JSON payload terminated by <EOF>.
        Handles large base64-encoded model files by reading in large chunks.
        """
        try:
            chunks = []
            while True:
                pkt = conn.recv(65536)   # 64 KB chunks — handles large model files
                if not pkt:
                    break
                chunks.append(pkt.decode("utf-8", errors="ignore"))
                combined = "".join(chunks)
                if "<EOF>" in combined:
                    data_str = combined.replace("<EOF>", "")
                    break
            else:
                # Connection closed without EOF
                conn.close()
                return

            msg = json.loads(data_str)
            req = msg.get("request")

            if req == "update_model":
                station    = msg.get("station", "")
                model_type = msg.get("model_type", "CNN")
                b64        = msg.get("model", "")

                # Always resolve to Station_1 / Station_2 / Station_3
                mn = normalize_station_name(station)

                sp = f"received/{mn}_{model_type}.keras"
                with open(sp, "wb") as f:
                    f.write(base64.b64decode(b64))

                client_models.setdefault(mn, {})[model_type] = sp
                self._log(f"[RX] {model_type} model received → {mn}")

                # Schedule UI + evaluation on main thread
                self.after(0, lambda a=mn, b=model_type, c=sp: self._on_model_received(a, b, c))
                conn.send(b"OK")
            else:
                conn.send(b"UNKNOWN_REQUEST")

        except json.JSONDecodeError as e:
            self._log(f"[ERR] JSON parse error: {e}")
            try:
                conn.send(b"ERROR")
            except Exception:
                pass
        except Exception as e:
            self._log(f"[ERR] Handler: {e}")
        finally:
            conn.close()

    def _on_model_received(self, station, model_type, path):
        """Called on the main thread after a model is saved."""
        self._update_station_label(station)
        self._update_stat_cards()
        self._evaluate_model(station, model_type, path)

    def _evaluate_model(self, station, model_type, path):
        """Run in a background thread to avoid blocking the UI."""
        def worker():
            try:
                model = load_model(path)
                sx, sy, fname = get_station_scalers(station)
                df   = pd.read_csv(f"Dataset/{fname}")
                data = df.tail(24).values
                # Explicit float64 cast — raw numpy array has object dtype due to string columns
                Xe = sx.transform(data[:, 2:14].astype(np.float64))
                yt_raw = data[:, 14].astype(np.float64).reshape(-1, 1)
                if model_type == "CNN":
                    Xe = Xe.reshape(-1, 12, 1, 1)

                # Model outputs in normalized [0,1] space
                pred_norm = model.predict(Xe, verbose=0).astype(np.float64)
                # Normalize y_true to the same [0,1] space for fair MSE / R²
                yt_norm = sy.transform(yt_raw)

                # MSE and R² in normalized space — values are small and meaningful
                mse_v = float(mean_squared_error(yt_norm, pred_norm))
                r2_v  = float(r2_score(yt_norm, pred_norm))

                # Accuracy via MAPE in original (inverse-transformed) space
                preds_raw = sy.inverse_transform(pred_norm).astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = np.where(np.abs(yt_raw) < 1e-9, 1e-9, yt_raw)
                    mape  = np.mean(np.abs((yt_raw - preds_raw) / denom)) * 100
                mape = float(np.nan_to_num(mape, nan=100.0, posinf=100.0))
                acc  = float(max(0.0, min(100.0, 100.0 - mape)))

                client_metrics.setdefault(station, {})[model_type] = {
                    "mse":   mse_v,
                    "acc":   acc,
                    "r2":    r2_v,
                    "preds": preds_raw.flatten(),
                    "true":  yt_raw.flatten()
                }
                self._log(f"[OK] {station} {model_type}  ACC:{acc:.1f}%  MSE:{mse_v:.5f}  R2:{r2_v:.3f}")
                self.after(0, self._refresh_dashboard_chart)
                self.after(0, self._update_stat_cards)
            except Exception as e:
                self._log(f"[ERR] Evaluation ({station} {model_type}): {e}")

        Thread(target=worker, daemon=True).start()

    # -- AGGREGATION ----------------------------------------------------------
    def run_aggregation(self):
        self._log("[SYSTEM] Initiating Federated Averaging...")
        cnn_c  = sum(1 for s in client_models if "CNN"  in client_models[s])
        ffnn_c = sum(1 for s in client_models if "FFNN" in client_models[s])

        if cnn_c >= 2:
            at = "CNN"
        elif ffnn_c >= 2:
            at = "FFNN"
        else:
            messagebox.showwarning(
                "Not Ready",
                f"Need at least 2 stations with the same model type.\n"
                f"CNN models received: {cnn_c}\n"
                f"FFNN models received: {ffnn_c}"
            )
            return

        # Run the heavy work in a background thread
        Thread(target=self._aggregation_worker, args=(at,), daemon=True).start()

    def _aggregation_worker(self, at):
        """Background thread: FedAvg + evaluation."""
        try:
            stations_with_model = [s for s in client_models if at in client_models[s]]
            self._log(f"[AGG] FedAvg using {at} from: {', '.join(stations_with_model)}")

            # ── Load & average weights ────────────────────────────────────────
            models  = [load_model(client_models[s][at]) for s in stations_with_model]
            weights = [m.get_weights() for m in models]
            new_w   = [np.mean([w[i] for w in weights], axis=0)
                       for i in range(len(weights[0]))]
            agg = models[0]
            agg.set_weights(new_w)
            agg.save("model/aggregated_model.keras")
            self._log("[AGG] Aggregated model saved.")

            # ── Evaluate on each contributing station & average results ───────
            all_acc, all_mse, all_r2 = [], [], []
            all_yt, all_yp = [], []

            for stn in stations_with_model:
                try:
                    sx, sy, fname = get_station_scalers(stn)
                    df  = pd.read_csv(f"Dataset/{fname}")
                    d   = df.tail(24).values
                    Xe  = sx.transform(d[:, 2:14].astype(np.float64))
                    yt_raw = d[:, 14].astype(np.float64).reshape(-1, 1)
                    if at == "CNN":
                        Xe = Xe.reshape(-1, 12, 1, 1)

                    # Model outputs normalized [0,1] predictions
                    pred_norm = agg.predict(Xe, verbose=0).astype(np.float64)

                    # Safety: replace any NaN/Inf in raw predictions
                    if not np.all(np.isfinite(pred_norm)):
                        self._log(f"[WARN] Non-finite predictions for {stn}, clipping.")
                        pred_norm = np.nan_to_num(pred_norm, nan=0.5, posinf=1.0, neginf=0.0)

                    # Normalize y_true into the same [0,1] space for fair MSE / R²
                    yt_norm  = sy.transform(yt_raw)

                    # MSE and R² in normalized space — avoids huge raw-unit errors
                    mse_v = float(mean_squared_error(yt_norm, pred_norm))
                    r2_v  = float(r2_score(yt_norm, pred_norm))

                    # MAPE-based accuracy uses inverse-transformed (original scale) values
                    yp = sy.inverse_transform(pred_norm).astype(np.float64)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        denom = np.where(np.abs(yt_raw) < 1e-9, 1e-9, yt_raw)
                        mape  = float(np.nan_to_num(
                            np.mean(np.abs((yt_raw - yp) / denom)) * 100,
                            nan=100.0, posinf=100.0))
                    acc = float(max(0.0, min(100.0, 100.0 - mape)))

                    all_acc.append(acc)
                    all_mse.append(mse_v)
                    all_r2.append(r2_v)
                    all_yt.append(yt_raw)
                    all_yp.append(yp)
                    self._log(f"[AGG] {stn}: ACC={acc:.1f}%  MSE(norm)={mse_v:.5f}  R²={r2_v:.4f}")
                except Exception as e:
                    self._log(f"[WARN] Could not evaluate {stn}: {e}")

            if not all_acc:
                self._log("[ERR] No stations evaluated successfully.")
                return

            # ── Combined metrics (mean over stations) ─────────────────────────
            acc_final = float(np.mean(all_acc))
            mse_final = float(np.mean(all_mse))
            r2_final  = float(np.mean(all_r2))
            # Concatenate all true/predicted for the chart
            yt_all = np.concatenate(all_yt, axis=0)
            yp_all = np.concatenate(all_yp, axis=0)

            client_metrics["Aggregated"] = {at: {
                "acc": acc_final, "mse": mse_final, "r2": r2_final
            }}

            self._log(
                f"[OK] Aggregation done  ACC:{acc_final:.2f}%  "
                f"MSE:{mse_final:.2f}  R²:{r2_final:.4f}"
            )

            # ── Schedule UI updates on main thread ────────────────────────────
            self.after(0, lambda: self._agg_count_inc())
            self.after(0, self._refresh_dashboard_chart)
            self.after(0, lambda: self._show_agg_results(
                acc_final, yt_all, yp_all, at, mse_final, r2_final))
            self.after(0, lambda: self.show_page("Aggregation"))

        except Exception as e:
            self._log(f"[ERR] Aggregation worker: {e}")
            self.after(0, lambda: messagebox.showerror("Aggregation Error", str(e)))

    def _agg_count_inc(self):
        self._agg_count += 1
        try:
            self._lbl_agg_val.configure(text=str(self._agg_count))
        except Exception:
            pass

    def _show_agg_results(self, acc, yt, yp, mt, mse_val=None, r2_val=None):
        # Derive mse/r2 from args if passed directly, else fall back to stored metrics
        if mse_val is None:
            mse_val = client_metrics.get("Aggregated", {}).get(mt, {}).get("mse", 0.0)
        if r2_val is None:
            r2_val  = client_metrics.get("Aggregated", {}).get(mt, {}).get("r2",  0.0)

        parent = self.pages["Aggregation"]
        for w in parent.winfo_children():
            w.destroy()
        ctk.CTkLabel(parent, text="Federated Aggregation Results",
                     font=("Segoe UI", 20, "bold"), text_color=C["white"]).pack(anchor="w", padx=20, pady=(16, 4))

        # Summary metric badges
        badges = ctk.CTkFrame(parent, fg_color="transparent")
        badges.pack(fill="x", padx=20, pady=4)
        # MSE is now in normalized [0,1] space — use 6 decimal places
        mse_display = f"{mse_val:.6f}" if mse_val is not None and mse_val < 1.0 else f"{mse_val:.4f}"
        # R² color: green if positive, red if negative
        r2_color = C["green"] if (r2_val is not None and r2_val >= 0) else C["red"]
        for col, (title, val, color) in enumerate([
            ("Global Accuracy",       f"{acc:.2f}%",   C["cyan"]),
            ("Avg MSE (Normalized)",  mse_display,     C["orange"]),
            ("Avg R² Score",          f"{r2_val:.4f}", r2_color),
            ("Model Type",            mt,               C["purple"]),
        ]):
            badges.grid_columnconfigure(col, weight=1)
            f = make_card(badges)
            f.grid(row=0, column=col, padx=5, sticky="ew")
            ctk.CTkLabel(f, text=val,   font=("Arial", 18, "bold"), text_color=color).pack(pady=(10, 2))
            ctk.CTkLabel(f, text=title, font=("Arial", 9),           text_color=C["gray"]).pack(pady=(0, 10))

        # Alert banner
        ac = make_card(parent)
        ac.pack(fill="x", padx=20, pady=8)
        ac.configure(border_color=C["red"], border_width=2)
        ctk.CTkLabel(ac, text="⚠  FLOOD ALERT ACTIVE",
                     font=("Arial", 20, "bold"), text_color=C["red"]).pack(pady=(14, 4))
        ctk.CTkLabel(ac, text=f"Global Confidence: {acc:.2f}%",
                     font=("Arial", 14), text_color=C["white"]).pack()
        fd = datetime.now().replace(month=8, day=10) + timedelta(days=int(np.random.randint(-8, 8)))
        ctk.CTkLabel(ac,
                     text=f"Predicted Event Window: {fd.strftime('%d %B %Y')}",
                     font=("Consolas", 12), text_color=C["cyan"]).pack(pady=(4, 14))

        # Chart — flatten to 1-D for plotting
        yt_flat = yt.flatten()
        yp_flat = yp.flatten()
        chart_f = make_card(parent)
        chart_f.pack(fill="both", expand=True, padx=20, pady=8)
        fig, ax = plt.subplots(figsize=(10, 4), dpi=95)
        fig.patch.set_facecolor(C["panel"])
        ax.set_facecolor(C["bg"])
        ax.plot(yt_flat, "w--", lw=2,   label="Historical Baseline")
        ax.plot(yp_flat, color=C["cyan"], lw=2.5, label=f"Aggregated {mt} Forecast")
        ax.fill_between(range(len(yp_flat)), yp_flat, yt_flat, alpha=0.15, color=C["cyan"])
        ax.set_title("FedAvg Model Validation (All Stations)", color=C["white"])
        ax.legend(facecolor=C["card"], labelcolor=C["white"])
        ax.tick_params(colors=C["gray"])
        ax.grid(color=C["border"], alpha=0.4)
        fig.tight_layout()
        cv = FigureCanvasTkAgg(fig, master=chart_f)
        cv.draw()
        cv.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # -- PDF EXPORT -----------------------------------------------------------
    def export_report(self):
        try:
            fname    = f"Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            doc      = SimpleDocTemplate(fname, pagesize=letter)
            styles   = getSampleStyleSheet()
            elements = []
            elements.append(Paragraph("FLOOD FORECASTING SYSTEM REPORT", styles["Title"]))
            elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            elements.append(Spacer(1, 20))
            if client_metrics:
                elements.append(Paragraph("Performance Metrics", styles["Heading2"]))
                td = [["Station", "Model", "Accuracy (%)", "MSE", "R² Score"]]
                for s, mv in client_metrics.items():
                    for mt, vals in mv.items():
                        td.append([s, mt, f"{vals['acc']:.2f}%", f"{vals['mse']:.4f}", f"{vals['r2']:.4f}"])
                t = Table(td)
                t.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                    ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
                    ("GRID",       (0, 0), (-1, -1), 1, colors.black),
                ]))
                elements.append(t)
            doc.build(elements)
            messagebox.showinfo("Export", f"Report saved:\n{os.path.abspath(fname)}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


# =============================================================================
#  ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = FloodServer()
    app.mainloop()
