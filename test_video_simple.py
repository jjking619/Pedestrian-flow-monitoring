#!/usr/bin/env python3
"""
行人流量监控 - 视频文件测试版本（简化版）
基于 ip_camera_main.py 的 BYTETracker + ReID 实现
只统计总检测人数，不包含进出计数功能
支持本地视频文件测试（如 street.mp4）
"""

import cv2
import numpy as np
import os
import sys
import argparse
from tracker.bytetrack import BYTETracker

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
):
    h, w = img.shape[:2]
    new_w, new_h = new_shape

    r = min(new_w / w, new_h / h)
    nw, nh = int(round(w * r)), int(round(h * r))

    img_resized = cv2.resize(img, (nw, nh))

    pad_w = new_w - nw
    pad_h = new_h - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img_padded, r, left, top

def yolo_v5_person_infer(
    frame,
    net,
    conf_thresh=0.25,
    iou_thresh=0.5,
    input_size=640
):
    """
    OpenCV DNN + YOLOv5n ONNX
    return: list of [x1, y1, x2, y2, score]
    """

    img, scale, pad_w, pad_h = letterbox(frame, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(
        img,
        scalefactor=1 / 255.0,
        size=(input_size, input_size),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    preds = net.forward()[0]   # shape: (25200, 85)

    h0, w0 = frame.shape[:2]
    boxes = []
    scores = []

    for det in preds:
        obj_conf = det[4]
        if obj_conf < conf_thresh:
            continue

        class_scores = det[5:]
        class_id = np.argmax(class_scores)

        # COCO: person == 0
        if class_id != 0:
            continue

        score = obj_conf * class_scores[class_id]
        if score < conf_thresh:
            continue

        cx, cy, w, h = det[:4]

        # 恢复到 letterbox 前
        x = (cx - w / 2 - pad_w) / scale
        y = (cy - h / 2 - pad_h) / scale
        w = w / scale
        h = h / scale

        x1 = max(0, min(int(x), w0 - 1))
        y1 = max(0, min(int(y), h0 - 1))
        x2 = max(0, min(int(x + w), w0 - 1))
        y2 = max(0, min(int(y + h), h0 - 1))

        # 新增：验证坐标有效性
        if x1 >= x2 or y1 >= y2:
            print(f"⚠️ 检测框坐标无效：x1={x1}, x2={x2}, y1={y1}, y2={y2}")
            continue

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(score))

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes,
        scores,
        conf_thresh,
        iou_thresh
    )

    results = []
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        results.append([x, y, x + w, y + h, scores[i]])

    print(f"✅ YOLOv5 检测到 {len(results)} 个人形目标")
    return results

def setup_video_capture(video_source):
    """Setup video capture from file or camera"""
    if isinstance(video_source, str) and os.path.exists(video_source):
        # Video file
        cap = cv2.VideoCapture(video_source)
        print(f"🎥 Loading video file: {video_source}")
    else:
        # Camera index
        cap = cv2.VideoCapture(int(video_source))
        print(f"🎥 Opening camera: {video_source}")
    
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {video_source}")
        sys.exit(1)
    
    return cap

def main():
    parser = argparse.ArgumentParser(description='Pedestrian Flow Monitoring with Video File (Simple Version)')
    parser.add_argument('--video', type=str, default='street.mp4', 
                       help='Path to video file (default: street.mp4)')
    parser.add_argument('--model', type=str, default='models/yolov5n_640.onnx',
                       help='Path to YOLOv5 ONNX model')
    args = parser.parse_args()

    # Load YOLOv5 model
    try:
        net = cv2.dnn.readNetFromONNX(args.model)
        print("✅ YOLOv5 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLOv5 model: {e}")
        print(f"Please ensure the model file exists at '{args.model}'")
        sys.exit(1)
    
    # Setup video capture
    cap = setup_video_capture(args.video)
    
    # Get actual frame dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    print(f"  Frame rate: {actual_fps:.2f} fps")
    
    # Initialize BYTETracker with ReID
    tracker = BYTETracker(
        track_thresh=0.5,      # 跟踪阈值
        high_thresh=0.5,       # 高置信度阈值
        low_thresh=0.1,        # 低置信度阈值（ByteTrack 的关键：利用低分检测框）
        match_thresh=0.65,      # 匹配阈值
        track_buffer=30,       # 跟踪缓冲区大小
        frame_rate=15,          # 帧率
        use_reid=True,         # 启用 ReID 特征
    )
    
    # Set window properties
    cv2.namedWindow("Pedestrian Flow Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pedestrian Flow Monitor", FRAME_WIDTH, FRAME_HEIGHT)
    
    frame_id = 0
    total_detected_count = 0
    print("\n🚀 Starting pedestrian flow monitoring with video file...")
    print("Press 'ESC' to exit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📹 End of video file reached")
                break

            # Run person detection
            persons = yolo_v5_person_infer(frame, net)
            
            # Update total count
            total_detected_count += len(persons)
            
            # Run tracking with ReID
            tracks = tracker.update(persons, frame=frame)
            
            # Draw detection boxes (green)
            for x1, y1, x2, y2, score in persons:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"det {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # # Draw tracking boxes (blue)
            # for x1, y1, x2, y2, track_id in tracks:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            #     cv2.putText(frame, f"ID:{track_id}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                
            # Display statistics
            cv2.putText(frame, f"Current frame detections: {len(persons)}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # cv2.putText(frame, f"Total detected so far: {total_detected_count}", (20, 70),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            # cv2.putText(frame, f"Active tracks: {len(tracks)}", (20, 110),
            #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow("Pedestrian Flow Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("🛑 Exit requested by user")
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    # Cleanup
    print(f"\n📊 Final Results:")
    print(f"   Total frames processed: {frame_id}")
    print(f"   Total detections: {total_detected_count}")
    print(f"   Average detections per frame: {total_detected_count/frame_id if frame_id > 0 else 0:.2f}")
    
    print("\n🧹 Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()