# Flood Forecasting Using Federated Learning

A distributed flood prediction system for the Krishna River Basin built with Python, Keras/TensorFlow, and federated learning principles. This repository contains a central server, client GUI, dataset files, trained models, and supporting assets.

## 🚀 Project Summary

This project demonstrates a privacy-preserving flood forecasting system where local monitoring stations train models on their own data and share only model weights with a central server. The central server aggregates client models using federated averaging to improve a global flood prediction model without sending raw hydrological data across the network.

## ⭐ Key Features

- Federated learning architecture with a central server and multiple client stations
- Local training for each station using private CSV data
- Support for FFNN and CNN model architectures
- GUI applications for both client and server
- Socket-based model transfer and aggregation
- Real-time training dashboards and reporting

## 📁 Repository Structure

- `Main.py` - Client application for local data preprocessing, model training, and upload
- `server1.py` - Central federated server application for receiving client model weights and aggregation
- `Server.py` - Additional server file included in the repository
- `Dataset/` - Input CSV datasets (`Station_1.csv`, `Station_2.csv`, `Station_3.csv`, and others)
- `model/` - Saved model weights and training artifacts
- `received/` - Model files received and stored by the server
- `assets/` - UI images, icons, and related assets
- `Project_Documentation.md` - Detailed project documentation and design notes
- `requirements.txt` - Python dependency list (verify before install)

## 🧰 Dependencies

Recommended Python version: `3.9+`

Core packages used by the project:

- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `Pillow`
- `customtkinter`
- `reportlab`
- `psutil`

## ⚙️ Setup

1. Open a terminal in this repository:
   ```bash
   cd /Users/sivareddythalapareddy/Desktop/Nxtwave/FloodForecasting
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify that the `Dataset/Station_*.csv` files exist.

## ▶️ Usage

### Start the central server

Run the server application first:

```bash
python server1.py
```

The server application launches a federated server GUI and listens for incoming client weight uploads.

### Start one or more client stations

Run the client application on each station machine or local instance:

```bash
python Main.py
```

The client application allows dataset loading, preprocessing, model training, evaluation, and uploading of trained weights to the server.

## 📌 Notes

- The server uses TCP socket communication to receive model data from clients.
- Trained models and received weights are stored under `model/` and `received/`.
- `Main.py` is the client GUI, while `server1.py` is the central federated aggregation server.
- If `requirements.txt` appears corrupted or incorrectly encoded, install the needed packages manually.

## 💡 Recommended Workflow

1. Start `server1.py` on the central machine.
2. Launch `Main.py` for each station client.
3. Train local models using station-specific datasets.
4. Upload client weights to the server.
5. Use the central server dashboard to aggregate and evaluate the global model.

## 📚 Documentation

For detailed architecture, design, and implementation notes, see `Project_Documentation.md`.

## 🧪 Additional Files

- `run.bat` and `runServer.bat` — Windows batch files for quick launching
- `arch_render.html` and `arch.md` — architecture documentation and diagrams
- `Individual.txt` — may contain individualized notes or project summary

## 🤝 Contributing

Contributions are welcome. To contribute, please:

1. Fork the repository
2. Create a new branch for your feature or fix
3. Test your changes locally
4. Submit a pull request with a clear description

## 📄 License

No license file is included in this repository. Add a `LICENSE` file if you want to define usage permissions.
