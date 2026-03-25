# 人流量统计设备

[中文](README_zh.md) | English

## 🎯 项目概述

本项目是一个运行在Quectel Pi H1智能主控板下的轻量级人流量统计设备，集成了目标检测、目标跟踪和行人重识别（ReID）技术，能够：

- 实时检测视频流中的人体目标
- 使用ByteTrack算法进行稳定的目标跟踪
- 基于ReID特征进行人员去重统计
- 支持USB摄像头、IP摄像头和本地视频文件输入
- 提供实时人数统计和累计去重人数统计

[界面预览]()

## ✨ 主要特性

### 核心功能
- **多源输入支持**：USB摄像头、ONVIF IP摄像头、本地视频文件
- **实时目标检测**：基于YOLOv5s ONNX模型，支持多种输入尺寸（320/416/640）
- **稳定目标跟踪**：集成ByteTrack算法，有效处理遮挡和目标丢失场景
- **智能人员统计**：
  - 实时人数统计（当前帧内的人数）
  - 累计去重人数统计（基于track_id的历史累计人数）
- **ReID增强**：可选启用OSNet ReID模型，提升跟踪稳定性
- **低延迟设计**：双线程架构，确保视频采集不被AI处理阻塞

### 技术优势
- **轻量级部署**：使用OpenCV DNN后端，无需PyTorch依赖
- **树莓派优化**：针对ARM设备进行性能优化
- **模块化设计**：各功能组件独立，便于维护和扩展
- **内存友好**：合理的队列大小和缓冲区设置

## 🏗️ 系统架构

```
人流量统计设备
├── 视频输入层
│   ├── USB摄像头 (usb_camera_main.py)
│   ├── IP摄像头 (ip_camera_main.py)  
│   └── 视频文件 (local_video_main.py)
├── AI处理层
│   ├── YOLOv5目标检测 (yolo_v5_person_infer)
│   ├── ByteTrack目标跟踪 (bytetrack.py)
│   └── OSNet ReID特征提取 (reid_extractor.py)
└── 统计输出层
    └── 人员计数器 (line_counter.py)
```

## 🔧 安装依赖

### 克隆代码
```

```

### Python依赖
```bash
# 安装项目依赖
pip3 install -r requirements.txt
```

**requirements.txt 内容：**
```txt
# Core computer vision and numerical libraries
opencv-python>=4.5.0
numpy>=1.19.0

# Scientific computing and optimization  
scipy>=1.7.0

# Performance optimization
numba>=0.64.0

# ONVIF camera discovery and control
wsdiscovery>=2.0.0
onvif-zeep>=0.3.0
```

## 🤖 模型准备

### 目标检测模型
项目支持以下YOLOv5s ONNX模型（位于 `models/` 目录）：

| 模型文件 | 输入尺寸 | 特点 |
|---------|---------|------|
| `yolov5s_320.onnx` | 320×320 | 速度最快，精度稍低 |
| `yolov5s_416.onnx` | 416×416 | 速度与精度平衡（默认）|
| `yolov5s_640.onnx` | 640×640 | 精度最高，速度较慢 |

### 行人重识别模型
- **ReID模型**：`osnet_x0_25_market1501.onnx`（需放置在 `src/` 目录）
- **输入尺寸**：256×128（宽×高）
- **特征维度**：512维归一化特征向量

> **注意**：ReID模型需要从Market1501等ReID数据集微调后的版本，不能直接使用ImageNet预训练模型。

## 🚀 使用方法

### USB摄像头模式

```bash
cd /home/pi/pedestrian-flow-monitoring/src
python3 usb_camera_main.py
```

**功能特点：**
- 自动检测可用的USB摄像头设备（ID 0-9）
- 自动设置640×480分辨率以优化树莓派性能
- 实时显示检测框和统计信息

### IP摄像头模式

```bash
cd /home/pi/pedestrian-flow-monitoring/src  
python3 ip_camera_main.py
```

**配置说明：**
编辑 `ip_camera_main.py` 文件中的摄像头连接参数：

```python
# 🔧 手动指定摄像头详情（替换为您的实际值）
HOST = "192.168.177.227"  # 摄像头IP地址
PORT = 80                 # HTTP端口，通常为80或8080
ONVIF_USER = ""           # 用户名（如果需要）
ONVIF_PASS = ""           # 密码（如果需要）
```

**功能特点：**
- 支持ONVIF协议自动发现摄像头Profile
- 自动选择子码流（sub-stream）以降低带宽消耗
- 使用TCP传输确保RTSP流稳定性

### 视频文件测试

```bash
cd /home/pi/pedestrian-flow-monitoring
python3 test_video_simple.py --video street.mp4
```

**命令行参数：**
- `--video`: 指定视频文件路径（默认使用 `street.mp4`）
- `--model`: 指定YOLO模型路径（可选）

## ⚙️ 配置说明

### YOLO检测参数
在 `usb_camera_main.py` 和 `ip_camera_main.py` 中可调整：

```python
# 检测置信度阈值（建议≥0.5以减少误检）
conf_thresh=0.5

# NMS IOU阈值（建议≥0.5以减少抖动）
iou_thresh=0.5

# YOLO输入尺寸（需与模型匹配）
input_size=416
```

### ByteTrack跟踪参数
```python
tracker = BYTETracker(
    track_thresh=0.5,      # 跟踪检测阈值
    high_thresh=0.5,       # 高置信度阈值
    low_thresh=0.1,        # 低置信度阈值（ByteTrack核心特性）
    match_thresh=0.7,      # 匹配阈值
    track_buffer=30,       # 跟踪缓冲区大小
    frame_rate=actual_fps, # 帧率
    use_reid=True,         # 启用ReID特征
)
```

### ReID融合权重
当启用ReID时，可调整IOU和特征的融合权重：

```python
# 在BYTETracker初始化中设置
iou_weight=0.3,    # IOU距离权重（建议≤0.3）
feat_weight=0.7,   # 特征距离权重（建议≥0.7）
```

## 🚀 性能优化

### 智能主控板专用优化
1. **分辨率选择**：USB摄像头默认使用640×480，平衡性能和精度
2. **模型选择**：推荐使用 `yolov5s_416.onnx` 或 `yolov5s_320.onnx`
3. **线程架构**：生产者-消费者模式，避免视频采集阻塞
4. **缓冲区设置**：`CAP_PROP_BUFFERSIZE=1` 减少延迟

### 内存管理
- 跟踪特征历史限制为50帧，防止内存膨胀
- 队列大小限制为2帧，避免内存堆积
- 及时清理已移除的跟踪目标

## ❓ 常见问题

### Q1: 摄像头无法打开
**解决方案：**
- 确认用户已添加到video组：`sudo usermod -aG video $USER`
- 重启系统使组权限生效
- 检查摄像头是否被其他程序占用

### Q2: 模型文件找不到
**解决方案：**
- 确保模型文件放在正确位置：
  - YOLO模型：`/home/pi/pedestrian-flow-monitoring/models/`
  - ReID模型：`/home/pi/pedestrian-flow-monitoring/src/`
- 修改代码中的模型路径变量

### Q3: IP摄像头连接失败
**解决方案：**
- 确认摄像头IP地址和端口正确
- 检查网络连通性：`ping 192.168.x.x`
- 确认ONVIF服务已启用
- 如需认证，填写正确的用户名和密码

### Q4: 性能卡顿
**解决方案：**
- 降低YOLO输入尺寸（使用320或416）
- 使用子码流（sub-stream）而非主码流
- 关闭ReID功能（`use_reid=False`）
- 降低显示窗口分辨率

## 📝 统计逻辑说明

### 两种计数类型
1. **实时计数（Current Count）**：当前帧检测到的人数
2. **累计计数（Total Count）**：基于track_id的历史累计去重人数

### 计数原理
- 每个新出现的track_id都会增加累计计数
- track_id由ByteTrack算法分配，具有唯一性
- 即使目标暂时丢失后重现，只要track_id相同就不会重复计数

## 🔄 退出程序

- 按下 **ESC** 键退出程序
- 程序会自动清理资源并关闭所有窗口

---