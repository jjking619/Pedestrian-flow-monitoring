# Pedestrian Flow Monitoring Device

[English](README.md) | [中文](README_zh.md)

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
| `yolov5n_320.onnx` | 320×320 | Fastest speed, slightly lower accuracy (default for USB/IP mode) |
| `yolov5n_416.onnx` | 416×416 | Balanced speed and accuracy (default for video file mode) |
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
- `--model`: Specify YOLO model path (optional, defaults to `yolov5n_416.onnx`)

**Examples:**
```bash
# Process video with default model
python3 local_video_main.py --video test_video.mp4

# Specify high-accuracy model
python3 local_video_main.py --video test_video.mp4 --model yolov5n_640.onnx
```

## ⚙️ Configuration Details

### YOLO Detection Parameters
Actual parameter values used in code:

```python
# Detection confidence threshold (actual value)
conf_thresh=0.25

# NMS IOU threshold (actual value)
iou_thresh=0.45

# YOLO input size
input_size=320  # Default for USB/IP mode
input_size=416  # Default for video file mode
```

### ByteTrack Tracking Parameters

**USB Camera and IP Camera Mode:**
```python
tracker = BYTETracker(
    track_thresh=0.2,      # Tracking detection threshold
    high_thresh=0.25,      # High confidence threshold
    low_thresh=0.05,       # Low confidence threshold (ByteTrack core feature)
    match_thresh=0.5,      # Matching threshold
    track_buffer=60,       # Tracking buffer size
    frame_rate=actual_fps, # Actual frame rate
    use_reid=True,         # Enable ReID features
    iou_weight=0.6,        # IOU distance weight
    feat_weight=0.3,       # Feature distance weight
)
```

**Local Video File Mode:**
```python
tracker = BYTETracker(
    track_thresh=0.2,      # Tracking detection threshold
    high_thresh=0.3,       # High confidence threshold
    low_thresh=0.05,       # Low confidence threshold
    match_thresh=0.5,      # Matching threshold
    track_buffer=30,       # Tracking buffer size
    frame_rate=30,         # Fixed frame rate
    use_reid=True,         # Enable ReID features
    iou_weight=0.6,        # IOU distance weight
    feat_weight=0.3,       # Feature distance weight
)
```

### Virtual Line Counting Configuration
- **Default Position**: Middle horizontal line (half of frame height)
- **Counting Logic**:
  - Downward crossing virtual line: Count as "In" (entering)
  - Upward crossing virtual line: Count as "Out" (exiting)
  - Each track_id is counted only once to prevent duplicate counting

### RTSP Stream Optimization (IP Camera Mode)
The system uses optimized FFmpeg parameters for RTSP streams:
```bash
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyzeduration;1000000|probesize;32"
```
- `rtsp_transport;tcp`: Ensures reliable transmission
- `fflags;nobuffer`: Disables decoder buffering
- `flags;low_delay`: Enables low latency mode
- `analyzeduration;1000000`: Reduces analysis time
- `probesize;32`: Minimizes probe data size
- `CAP_PROP_BUFFERSIZE=1`: Minimizes capture buffer size

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

## 🚀 Performance Optimization

### Intelligent Main Control Board Optimization
1. **Resolution Selection**: USB camera defaults to 640×480, balancing performance and accuracy
2. **Model Selection**: Recommended to use `yolov5n_320.onnx` or `yolov5n_416.onnx`
3. **Thread Architecture**: Producer-consumer pattern to avoid video capture blocking
4. **Buffer Settings**: `CAP_PROP_BUFFERSIZE=1` to reduce latency
5. **RTSP Optimization**: IP camera uses TCP transport + low-latency FFmpeg parameters

### Memory Management
- Track feature history limited to 50 frames to prevent memory bloat
- Queue size limits (USB mode: 5 frames, IP/video mode: 2 frames) to avoid memory accumulation
- Timely cleanup of removed tracking targets

## 🔄 Exit Program

- Press **ESC** key to exit the program
- The program will automatically clean up resources and close all windows

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