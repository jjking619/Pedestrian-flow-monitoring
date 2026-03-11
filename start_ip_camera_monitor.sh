#!/bin/bash

# IP Camera Pedestrian Flow Monitoring Startup Script

echo "🚀 Starting IP Camera Pedestrian Flow Monitoring System"

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "❌ Error: 'models' directory not found!"
    echo "Please ensure you have the YOLOv5 model file in the 'models' directory."
    echo "Expected file: models/yolov5n_person.onnx"
    exit 1
fi

# Check if model file exists
if [ ! -f "models/yolov5n_person.onnx" ]; then
    echo "❌ Error: YOLOv5 model file not found!"
    echo "Please download the model file to: models/yolov5n_person.onnx"
    exit 1
fi

# Check if required Python packages are installed
python3 -c "import cv2, numpy, onvif, wsdiscovery" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Required Python packages not found. Installing dependencies..."
    pip3 install opencv-python numpy onvif-zeep wsdiscovery
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install required packages. Please install manually:"
        echo "pip3 install opencv-python numpy onvif-zeek wsdiscovery"
        exit 1
    fi
fi

# Run the IP camera monitoring system
echo "🎥 Starting IP camera discovery and monitoring..."
python3 ip_camera_main.py

echo "👋 IP Camera monitoring stopped."