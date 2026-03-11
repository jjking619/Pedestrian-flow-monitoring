#!/usr/bin/env python3
"""
Simple RTSP Pedestrian Flow Monitoring
Directly use RTSP URL for pedestrian counting without ONVIF discovery
"""

import cv2
import numpy as np
import sys
from tracker.sort import Sort
from counter.line_counter import LineCounter

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

tracker = Sort()
counter = LineCounter(line_y=0)

def letterbox(
    img,
    new_shape=(360, 240),
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
    conf_thresh=0.4,
    iou_thresh=0.45,
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
    preds = net.forward()[0]

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

    return results

def setup_rtsp_stream(rtsp_url):
    """Setup RTSP stream with UDP transport for better performance and lower latency"""
    import os
    # Set environment variable to force UDP transport (faster but less reliable than TCP)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|buffer_size;1024"
    
    cap = cv2.VideoCapture(rtsp_url)
    
    # Set shorter timeouts for UDP
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 2000)  # Reduced timeout for UDP
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)  # Reduced timeout for UDP
    
    # Critical: Minimal buffer to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to set reasonable resolution to reduce processing load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

def main():
    if len(sys.argv) != 2:
        print("Usage: python rtsp_pedestrian_monitor.py <RTSP_URL>")
        print("Example: python rtsp_pedestrian_monitor.py rtsp://192.168.1.100:554/stream1")
        sys.exit(1)
    
    rtsp_url = sys.argv[1]
    print(f"📹 Starting pedestrian flow monitoring for RTSP stream: {rtsp_url}")
    
    # Load YOLOv5 model
    try:
        net = cv2.dnn.readNetFromONNX("models/yolov5n_person.onnx")
        print("✅ YOLOv5 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLOv5 model: {e}")
        print("Please ensure the model file exists at 'models/yolov5n_person.onnx'")
        sys.exit(1)
    
    # Setup video capture
    print("🎥 Setting up RTSP stream...")
    cap = setup_rtsp_stream(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Failed to open RTSP stream")
        sys.exit(1)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    print(f"  Frame rate: {actual_fps:.2f} fps")
    
    # Start processing loop
    print("\n🚀 Starting pedestrian flow monitoring...")
    print("Press 'ESC' to exit")
    
    cv2.namedWindow("RTSP Pedestrian Flow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RTSP Pedestrian Flow", FRAME_WIDTH, FRAME_HEIGHT)
    
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame from RTSP stream")
            break
        
        # Update counting line position (middle of frame)
        LINE_Y = frame.shape[0] // 2
        counter.set_line_y(LINE_Y)
        
        # Run person detection
        persons = yolo_v5_person_infer(frame, net)
        tracks = tracker.update(persons)
        counter.update(tracks)
        
        in_count, out_count = counter.get_counts()
        total_count = counter.total_count
        
        # Visualization - 使用更小的字体
        for x1, y1, x2, y2, score in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"person {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # 减小字体大小
                (0, 255, 0),
                1  # 减小字体粗细
            )
        
        # Draw counting line
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)
        
        # Display statistics - 使用更小的字体
        cv2.putText(frame, f"Total count: {total_count}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"IN: {in_count}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"OUT: {out_count}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("RTSP Pedestrian Flow", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        
        frame_id += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Pedestrian flow monitoring stopped.")

if __name__ == "__main__":
    main()