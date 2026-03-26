# 人流量统计设备

[中文](README_zh.md) | [English](README.md)

## 🎯 项目概述

本项目是一个运行在Quectel Pi H1智能主控板下的轻量级人流量统计设备，集成了目标检测、目标跟踪和行人重识别（ReID）技术，能够：

- 实时检测视频流中的人体目标
- 使用ByteTrack算法进行稳定的目标跟踪
- 基于ReID特征进行人员去重统计
- 支持USB摄像头、IP摄像头、本地视频文件
- 提供实时人数统计、累计去重人数统计和进出方向统计

[界面预览]()

## ✨ 主要特性

### 核心功能
- **多源输入支持**：USB摄像头、ONVIF IP摄像头、本地视频文件
- **实时目标检测**：基于YOLOv5n ONNX模型，支持多种输入尺寸（320/416/640）
- **稳定目标跟踪**：集成ByteTrack算法，有效处理遮挡和目标丢失场景
- **智能人员统计**：
  - 实时人数统计（当前帧内的人数）
  - 累计去重人数统计（基于track_id的历史累计人数）
  - 进出方向统计（基于虚拟线的流向分析）
- **ReID增强**：可选启用OSNet ReID模型，提升跟踪稳定性


## 🏗️ 系统架构

```
人流量统计设备
├── 视频输入层
│   ├── USB摄像头 (usb_camera_main.py)
│   ├── IP摄像头 (ip_camera_main.py)  
│   └── 本地视频文件 (local_video_main.py)
├── AI处理层
│   ├── YOLOv5n目标检测 (yolo_v5_person_infer)
│   ├── ByteTrack目标跟踪 (bytetrack.py)
│   └── OSNet ReID特征提取 (reid_extractor.py)
└── 统计输出层
    └── 人员计数器 (line_counter.py)
```

## 🔧 安装依赖

### 克隆代码
```bash
git clone <repository-url>
cd demo-people-counting-device/
```

### Python依赖
```bash
# 安装项目依赖
pip3 install -r requirements.txt
```

## 🤖 模型准备

### 目标检测模型
项目支持以下YOLOv5n ONNX模型（位于 `src/` 目录）：

| 模型文件 | 输入尺寸 | 特点 |
|---------|---------|------|
| `yolov5n_320.onnx` | 320×320 | 速度最快，精度稍低（USB/IP模式默认）|
| `yolov5n_416.onnx` | 416×416 | 速度与精度平衡（本地视频文件测试模式默认）|
| `yolov5n_640.onnx` | 640×640 | 精度最高，速度较慢 |

> **注意**：所有模型文件已包含在项目中，位于 `src/` 目录下，无需额外下载。

### 行人重识别模型
- **ReID模型**：`osnet_x0_25_market1501.onnx`（位于 `src/` 目录）
- **输入尺寸**：256×128（宽×高）
- **特征维度**：512维归一化特征向量

> **注意**：ReID模型需要从Market1501等ReID数据集微调后的版本，不能直接使用ImageNet预训练模型。

## 🚀 使用方法

### USB摄像头模式

```bash
cd ~/demo-people-counting-device/src
python3 usb_camera_main.py
```

### IP摄像头模式

```bash
cd ~/demo-people-counting-device/src  
python3 ip_camera_main.py
```

### 本地视频文件测试

```bash
cd ~/demo-people-counting-device/src
python3 local_video_main.py --video ../asset/street.mp4
```

**命令行参数：**
- `--video`: 指定视频文件路径（必填）
- `--model`: 指定YOLO模型路径（可选，默认使用 `yolov5n_416.onnx`）

**示例：**
```bash
# 使用默认模型处理视频
python3 local_video_main.py --video test_video.mp4

# 指定高精度模型
python3 local_video_main.py --video test_video.mp4 --model yolov5n_640.onnx
```



## ⚙️ 配置详情

### YOLO检测参数
代码中实际使用的参数值：

```python
# 检测置信度阈值（实际值）
conf_thresh=0.25

# NMS IOU阈值（实际值）
iou_thresh=0.45

# YOLO输入尺寸
input_size=320  # USB/IP模式默认值
input_size=416  # 视频文件模式默认值
```

### ByteTrack跟踪参数

**USB摄像头和IP摄像头模式：**
```python
tracker = BYTETracker(
    track_thresh=0.2,      # 跟踪检测阈值
    high_thresh=0.25,      # 高置信度阈值
    low_thresh=0.05,       # 低置信度阈值（ByteTrack核心特性）
    match_thresh=0.5,      # 匹配阈值
    track_buffer=60,       # 跟踪缓冲区大小
    frame_rate=actual_fps, # 实际帧率
    use_reid=True,         # 启用ReID特征
    iou_weight=0.6,        # IOU距离权重
    feat_weight=0.3,       # 特征距离权重
)
```

**本地视频文件模式：**
```python
tracker = BYTETracker(
    track_thresh=0.2,      # 跟踪检测阈值
    high_thresh=0.3,       # 高置信度阈值
    low_thresh=0.05,       # 低置信度阈值
    match_thresh=0.5,      # 匹配阈值
    track_buffer=30,       # 跟踪缓冲区大小
    frame_rate=30,         # 固定帧率
    use_reid=True,         # 启用ReID特征
    iou_weight=0.6,        # IOU距离权重
    feat_weight=0.3,       # 特征距离权重
)
```

### 虚拟线计数配置
- **默认位置**：画面中间水平线（画面高度的一半）
- **计数逻辑**：
  - 向下穿越虚拟线：计入"In"（进入）
  - 向上穿越虚拟线：计入"Out"（离开）
  - 每个track_id仅计数一次，防止重复统计

### RTSP流优化（IP摄像头模式）
系统使用优化的FFmpeg参数处理RTSP流：
```bash
OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyzeduration;1000000|probesize;32"
```
- `rtsp_transport;tcp`：确保可靠传输
- `fflags;nobuffer`：禁用解码器缓冲
- `flags;low_delay`：启用低延迟模式
- `analyzeduration;1000000`：减少分析时间
- `probesize;32`：最小化探测数据量
- `CAP_PROP_BUFFERSIZE=1`：最小化采集缓冲区大小

## 📝 统计逻辑说明

### 三种计数类型
1. **实时计数（Current Count）**：当前帧检测到的活跃人数
2. **累计计数（Total Count）**：基于track_id的历史累计去重人数
3. **进出计数（In/Out Count）**：基于虚拟线的进出方向统计

### 计数原理
- **实时计数**：直接统计当前帧中活跃的track数量
- **累计计数**：每个新出现的track_id都会增加累计计数，track_id由ByteTrack算法分配，具有唯一性
- **进出计数**：通过虚拟线（默认画面中间水平线）检测目标跨越方向：
  - 向下移动（y坐标增大）：计入"In"
  - 向上移动（y坐标减小）：计入"Out"
  - 使用目标中心点的历史轨迹判断穿越方向
  - 每个track_id只会被计数一次，防止重复统计

### 虚拟线自定义
当前版本使用默认中间线，支持自定义虚拟线位置和方向：
- **水平线**：`direction='horizontal'`，`line_position=指定Y坐标`
- **垂直线**：`direction='vertical'`，`line_position=指定X坐标`

## ❓ 常见问题

### Q1: 摄像头无法打开
**解决方案：**
- 确认用户已添加到video组：`sudo usermod -aG video $USER`
- 重启系统使组权限生效
- 检查摄像头是否被其他程序占用

### Q2: 模型文件找不到
**解决方案：**
- 确保在 `src/` 目录下运行脚本（所有模型文件都在此目录）
- 不要修改工作目录，直接在 `src/` 目录下执行命令

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
- 关闭ReID功能（修改代码中 `use_reid=False`）
- 降低显示窗口分辨率

## 报告问题
欢迎提交Issue和Pull Request来改进此项目。
