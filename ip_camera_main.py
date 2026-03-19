#!/usr/bin/env python3
import cv2
import numpy as np
import os
import sys
import threading
import queue
import traceback
from urllib.parse import urlparse
from tracker.bytetrack import BYTETracker  

from wsdiscovery import WSDiscovery
from onvif import ONVIFCamera

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Default authentication (can be modified as needed)
ONVIF_USER = ""
ONVIF_PASS = ""

# Global variables for thread communication
frame_queue = queue.Queue(maxsize=2)  # 增加队列大小，减少丢帧
result_queue = queue.Queue(maxsize=1)  # Store latest processing result
stop_event = threading.Event()
processing_lock = threading.Lock()

def discover_onvif_devices(timeout=3):
    """WS-Discovery search for ONVIF devices in local network"""
    wsd = WSDiscovery()
    wsd.start()
    services = wsd.searchServices(timeout=timeout)
    devices = []
    for svc in services:
        for addr in svc.getXAddrs():
            if 'onvif' in addr.lower():
                devices.append(addr)
                break
    wsd.stop()
    return devices

def get_profile_info(cam, profile):
    """Extract resolution, RTSP address and other info from a single Profile"""
    token = profile.token
    name = getattr(profile, 'Name', token)
    
    # Get resolution
    width = height = None
    if hasattr(profile, 'VideoEncoderConfiguration') and profile.VideoEncoderConfiguration:
        resolution = profile.VideoEncoderConfiguration.Resolution
        width = resolution.Width
        height = resolution.Height
    
    # Get RTSP stream address
    media = cam.create_media_service()
    try:
        stream_uri = media.GetStreamUri({
            'StreamSetup': {
                'Stream': 'RTP-Unicast',
                'Transport': {'Protocol': 'RTSP'}
            },
            'ProfileToken': token
        })
        rtsp_url = stream_uri.Uri
    except Exception as e:
        print(f"    Failed to get Profile {token} stream address: {e}")
        return None
    
    return {
        'token': token,
        'name': name,
        'width': width,
        'height': height,
        'rtsp_url': rtsp_url
    }

def get_all_profiles(host, port, user, passwd):
    """Connect to device and get all available Profiles with detailed info"""
    try:
        cam = ONVIFCamera(host, port, user, passwd)
        
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            print("  Device has no available Profiles")
            return None
        
        profile_list = []
        for p in profiles:
            info = get_profile_info(cam, p)
            if info:
                # Complete authentication info (if not included in URL)
                if user and passwd and '@' not in info['rtsp_url']:
                    parsed = urlparse(info['rtsp_url'])
                    auth_url = f"{parsed.scheme}://{user}:{passwd}@{parsed.netloc}{parsed.path}"
                    if parsed.query:
                        auth_url += f"?{parsed.query}"
                    if parsed.fragment:
                        auth_url += f"#{parsed.fragment}"
                    info['rtsp_url'] = auth_url
                profile_list.append(info)
        
        return profile_list
    except Exception as e:
        print(f"  Failed to connect to device: {e}")
        return None

def select_main_sub(profiles):
    """
    Distinguish main stream and sub-stream from profiles list.
    Returns (main, sub):
      - main: Profile with highest resolution
      - sub:  Profile with second highest resolution (None if not available)
    """
    if not profiles:
        return None, None
    
    # Filter out Profiles without resolution (usually present)
    valid = [p for p in profiles if p['width'] and p['height']]
    if not valid:
        # If no resolution info, take first two by list order
        valid = profiles
    
    # Sort by resolution descending (width*height)
    sorted_profiles = sorted(valid, key=lambda p: (p['width'] or 0) * (p['height'] or 0), reverse=True)
    
    main = sorted_profiles[0] if sorted_profiles else None
    sub = sorted_profiles[1] if len(sorted_profiles) > 1 else None
    return main, sub

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
    conf_thresh=0.3,
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
    """Setup RTSP stream with TCP transport for better reliability"""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|analyzeduration;1000000|probesize;32"
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def ai_processing_worker(net,actual_fps):
    """Worker thread for AI processing and tracking"""
    # 使用 ByteTrack
    tracker = BYTETracker(
        track_thresh=0.5,      # 跟踪阈值
        high_thresh=0.5,       # 高置信度阈值
        low_thresh=0.1,        # 低置信度阈值（ByteTrack 的关键：利用低分检测框）
        match_thresh=0.8,      # 匹配阈值
        track_buffer=30,       # 跟踪缓冲区大小
        frame_rate=actual_fps,          # 帧率
        use_reid=True,         # 启用 ReID 特征
    )
    
    while not stop_event.is_set():
        try:
            # 获取最新帧 - 清空队列中的旧帧，只处理最新的一帧
            frame_data = None
            while not frame_queue.empty():
                try:
                    frame_data = frame_queue.get_nowait()
                    frame_queue.task_done()
                except queue.Empty:
                    break
            
            if frame_data is None:
                # 如果队列为空，等待新帧
                frame_data = frame_queue.get(timeout=1.0)
                frame_queue.task_done()
                
            if frame_data is None:
                break
                
            frame, frame_id = frame_data
                
            # Run person detection
            persons = yolo_v5_person_infer(frame, net,)
                
            # 直接调用tracker.update()，传入frame供内部ReID特征提取
            tracks = tracker.update(persons, frame=frame)
            
            # 统计总人数
            total_count = len(persons)
            
            # Put results in result queue (overwrite old results if queue is full)
            try:
                result_queue.put_nowait({
                    'frame': frame,
                    'persons': persons,
                    'tracks': tracks,
                    'total_count': total_count,
                    'frame_id': frame_id
                })
            except queue.Full:
                # Remove old result and add new one
                try:
                    result_queue.get_nowait()
                    result_queue.put_nowait({
                        'frame': frame,
                        'persons': persons,
                        'tracks': tracks,
                        'total_count': total_count,
                        'frame_id': frame_id
                    })
                except queue.Empty:
                    pass
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"AI processing error: {e}")
            traceback.print_exc()

def main():

    # Step 1: Discover ONVIF devices
    print("\n🔍 Searching for ONVIF devices...")
    devices = discover_onvif_devices(timeout=5)
    
    if not devices:
        print("❌ No ONVIF devices found. Please check your network connection.")
        sys.exit(1)
    
    print(f"✅ Found {len(devices)} ONVIF device(s):")
    for i, url in enumerate(devices, 1):
        print(f"  [{i}] {url}")
    
    # Step 2: Connect to the first device and get profiles
    dev_url = devices[0]
    print(f"\n📡 Connecting to device: {dev_url}")
    parsed = urlparse(dev_url)
    host = parsed.hostname
    port = parsed.port or 80
    
    profiles = get_all_profiles(host, port, ONVIF_USER, ONVIF_PASS)
    if not profiles:
        print("❌ Cannot get Profile info from the device.")
        sys.exit(1)
    
    print(f"  Got {len(profiles)} Profiles:")
    for p in profiles:
        res = f"{p['width']}x{p['height']}" if p['width'] and p['height'] else "Unknown resolution"
        print(f"    - {p['name']} ({p['token']}): {res}")
    
    # Step 3: Select main/sub streams
    main, sub = select_main_sub(profiles)
    if not main:
        print("❌ No valid main stream found.")
        sys.exit(1)
    
    print(f"  ✅ Main stream: {main['name']} ({main['width']}x{main['height']})")
    if sub:
        print(f"  ✅ Sub-stream: {sub['name']} ({sub['width']}x{sub['height']})")
        selected_stream = sub  
    else:
        print("  ⚠️ Only one stream available, using main stream")
        selected_stream = main
    rtsp_url = selected_stream['rtsp_url']
    print(f"  📺 Using RTSP URL: {rtsp_url}")
    
    # Step 4: Load YOLOv5 model
    try:
        net = cv2.dnn.readNetFromONNX("models/yolov5n_640.onnx")
        print("✅ YOLOv5 model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load YOLOv5 model: {e}")
        print("Please ensure the model file exists at 'models/yolov5n_416.onnx'")
        sys.exit(1)
    
    # Step 5: Setup video capture
    print("\n🎥 Setting up RTSP stream...")
    cap = setup_rtsp_stream(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Failed to open RTSP stream")
        sys.exit(1)
    
    # Get actual frame dimensions
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Frame dimensions: {actual_width}x{actual_height}")
    print(f"  Frame rate: {actual_fps:.2f} fps")
    
    # Step 6: Start AI processing worker thread
    print("\n🧵 Starting AI processing worker thread...")
    worker_thread = threading.Thread(target=ai_processing_worker, args=(net,actual_fps))
    worker_thread.daemon = True
    worker_thread.start()
    
    # Step 7: Start main processing loop (frame capture)
    print("\n🚀 Starting pedestrian flow monitoring...")
    print("Press 'ESC' to exit")
    
    # Set window properties
    cv2.namedWindow("Pedestrian Flow Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pedestrian Flow Monitor", FRAME_WIDTH, FRAME_HEIGHT)
    
    frame_id = 0
    last_processed_frame_id = -1
    last_display_frame = None  # 缓存上一次显示的带标注帧
    startup_phase = True  # 启动阶段标志

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame from RTSP stream")
                break

            # 确保队列中始终有最新帧
            try:
                frame_queue.put((frame.copy(), frame_id), block=False)
            except queue.Full:
                try:
                    frame_queue.get(block=False)
                    frame_queue.task_done()
                except queue.Empty:
                    pass
                try:
                    frame_queue.put((frame.copy(), frame_id), block=False)
                except queue.Full:
                    pass

            # 获取最新处理结果
            result = None
            try:
                # 清空旧的结果，只保留最新的
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()
                        result_queue.task_done()
                    except queue.Empty:
                        break
                if result is None and not result_queue.empty():
                    result = result_queue.get_nowait()
                    result_queue.task_done()
            except queue.Empty:
                pass

            # 显示逻辑优化
            if result is not None and result['frame_id'] >= last_processed_frame_id:
                # 构建带标注的显示帧
                display_frame = result['frame'].copy()
                persons = result['persons']
                total_count = result['total_count']
                current_frame_id = result['frame_id']
                
                # 更新最后处理的帧ID
                last_processed_frame_id = current_frame_id

                # 绘制检测框
                for x1, y1, x2, y2, score in persons:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        f"person {score:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1
                    )
                # 显示总人数统计信息
                cv2.putText(display_frame, f"Total Count: {total_count}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # 更新缓存并显示
                last_display_frame = display_frame.copy()
                cv2.imshow("Pedestrian Flow Monitor", display_frame)
                startup_phase = False  # 启动阶段结束
                
            else:
                # 启动阶段或没有新结果时，显示当前原始帧（避免显示旧的处理结果）
                if startup_phase:
                    # 启动阶段显示原始帧，避免显示初始化时的旧帧
                    cv2.imshow("Pedestrian Flow Monitor", frame)
                else:
                    # 非启动阶段，如果有助化帧则显示，否则显示当前帧
                    if last_display_frame is not None:
                        cv2.imshow("Pedestrian Flow Monitor", last_display_frame)
                    else:
                        cv2.imshow("Pedestrian Flow Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("🛑 Exit requested by user")
                stop_event.set()
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        stop_event.set()
    
    # Cleanup
    print("\n🧹 Cleaning up resources...")
    stop_event.set()
    
    # Wait for worker thread to finish
    if worker_thread.is_alive():
        worker_thread.join(timeout=2.0)
    
    # Clear queues
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
            frame_queue.task_done()
        except queue.Empty:
            break
    
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
            result_queue.task_done()
        except queue.Empty:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()