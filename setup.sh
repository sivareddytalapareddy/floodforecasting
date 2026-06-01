#!/bin/bash
# FloodForecasting Setup Script
# Run this script to install all dependencies and launch the application

echo "============================================"
echo "  Flood Forecasting Model - Setup Script"
echo "============================================"
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python is not installed. Please install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✅ Found Python: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"

# Install dependencies
echo ""
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "  ✅ Setup Complete!"
    echo "============================================"
    echo ""
    echo "To run the Client:  python Main.py"
    echo "To run the Server:  python Server.py"
    echo ""
    echo "Starting the Client now..."
    echo ""
    python Main.py
else
    echo ""
    echo "❌ Failed to install dependencies. Please check your Python version (3.10+ recommended)."
    exit 1
fi
