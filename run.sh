#!/bin/bash

# CropPrediction Project Startup Script
echo "================================"
echo "CropPrediction - Running Project"
echo "================================"

# Navigate to backend directory
cd "$(dirname "$0")/backend" || exit 1

# Install dependencies if needed
echo "Installing dependencies..."
pip install -r requirements.txt

# Start the Flask application
echo ""
echo "Starting Flask application..."
echo "==============================="
python app.py
