# Pedestrian Flow Monitoring Device

[English]| [中文](README_zh.md)

## 🎯 Project Overview

This project is a lightweight pedestrian flow monitoring device running on Quectel Pi H1 Smart Single-Board Computer, integrating object detection, object tracking, and person re-identification (ReID) technologies. It can:

- Real-time detect human targets in video streams
- Perform stable object tracking using ByteTrack algorithm
- Conduct person deduplication counting based on ReID features
- Support USB cameras, IP cameras, and local video files
- Provide real-time counting, cumulative deduplicated counting, and in/out direction statistics

[Interface Preview]()

## ✨ Key Features

### Core Functions
- **Multi-source Input Support**: USB cameras, ONVIF IP cameras, local video files
- **Real-time Object Detection**: Based on YOLOv5n ONNX model, supporting multiple input sizes (320/416/640)
- **Stable Object Tracking**: Integrated ByteTrack algorithm, effectively handling occlusion and target loss scenarios
- **Intelligent Person Counting**:
  - Real-time counting (number of people in current frame)
  - Cumulative deduplicated counting (historical cumulative count based on track_id)
  - In/out direction counting (flow analysis based on virtual line)
- **ReID Enhancement**: Optional OSNet ReID model to improve tracking stability


## 🏗️ System Architecture

```
Pedestrian Flow Monitoring Device
├── Video Input Layer
│   ├── USB Camera (usb_camera_main.py)
│   ├── IP Camera (ip_camera_main.py)  
│   └── Local Video File (local_video_main.py)
├── AI Processing Layer
│   ├── YOLOv5n Object Detection (yolo_v5_person_infer)
│   ├── ByteTrack Object Tracking (bytetrack.py)
│   └── OSNet ReID Feature Extraction (reid_extractor.py)
└── Statistics Output Layer
    └── Person Counter (line_counter.py)
```

## 🔧 Installation Dependencies

### Clone Repository
```bash
git clone <repository-url>
cd demo-people-counting-device/
```

### Python Dependencies
```bash
# Install project dependencies
pip3 install -r requirements.txt
```


## 🤖 Model Preparation

### Object Detection Models
The project supports the following YOLOv5n ONNX models (located in `src/` directory):

| Model File | Input Size | Features |
|-----------|------------|----------|
| `yolov5n_320.onnx` | 320×320 | Fastest speed, slightly lower accuracy (default  mode) |
| `yolov5n_416.onnx` | 416×416 | Balanced speed and accuracy |
| `yolov5n_640.onnx` | 640×640 | Highest accuracy, slower speed |

> **Note**: All model files are included in the project and located in the `src/` directory, no additional download required.

### Person Re-identification Model
- **ReID Model**: `osnet_x0_25_market1501.onnx` (located in `src/` directory)
- **Input Size**: 256×128 (width×height)
- **Feature Dimension**: 512-dimensional normalized feature vector

> **Note**: The ReID model requires fine-tuning from ReID datasets like Market1501, and cannot directly use ImageNet pre-trained models.

## 🚀 Usage Instructions

### USB Camera Mode

```bash
cd ~/demo-people-counting-device/src
python3 usb_camera_main.py
```

### IP Camera Mode

```bash
cd ~/demo-people-counting-device/src  
python3 ip_camera_main.p
```

### Local Video File Testing

```bash
cd ~/demo-people-counting-device/src
python3 local_video_main.py --video ../asset/street.mp4
```

**Command-line Arguments:**
- `--video`: Specify video file path (required)
- `--model`: Specify YOLO model path (optional, defaults to `yolov5n_320.onnx`)

**Examples:**
```bash
# Process video with default model
python3 local_video_main.py --video test_video.mp4

# Specify high-accuracy model
python3 local_video_main.py --video test_video.mp4 --model yolov5n_640.onnx
```

## 📝 Counting Logic Explanation

### Three Counting Types
1. **Real-time Count**: Active people count in current frame
2. **Cumulative Count**: Historical cumulative deduplicated count based on track_id
3. **In/Out Count**: Direction-based counting based on virtual line

### Counting Principles
- **Real-time Count**: Directly counts active tracks in current frame
- **Cumulative Count**: Each new track_id increases cumulative count; track_id assigned by ByteTrack algorithm is unique
- **In/Out Count**: Detects target crossing direction through virtual line (default middle horizontal line):
  - Downward movement (increasing y-coordinate): Count as "In"
  - Upward movement (decreasing y-coordinate): Count as "Out"
  - Uses target center point historical trajectory to determine crossing direction
  - Each track_id is counted only once to prevent duplicate counting

### Virtual Line Customization
Although the current version uses default middle line, the `LineCounter` class supports custom virtual line position and direction:
- **Horizontal Line**: `direction='horizontal'`, `line_position=specified Y coordinate`
- **Vertical Line**: `direction='vertical'`, `line_position=specified X coordinate`


## ❓ Common Issues

### Q1: Camera Cannot Be Opened
**Solutions:**
- Ensure user is added to video group: `sudo usermod -aG video $USER`
- Restart system to apply group permissions
- Check if camera is occupied by other programs

### Q2: Model Files Not Found
**Solutions:**
- Ensure running scripts from `src/` directory (all model files are located here)
- Do not change working directory, execute commands directly in `src/` directory

### Q3: IP Camera Connection Failed
**Solutions:**
- Verify camera IP address and port are correct
- Check network connectivity: `ping 192.168.x.x`
- Confirm ONVIF service is enabled
- Fill in correct username and password if authentication is required

### Q4: Performance Lag
**Solutions:**
- Reduce YOLO input size (use 320 or 416)
- Use sub-stream instead of main stream
- Disable ReID feature (set `use_reid=False` in code)
- Reduce display window resolution

## Reporting Issues
We welcome Issues and Pull Requests to improve this project.