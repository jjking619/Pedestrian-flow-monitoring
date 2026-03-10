import cv2
import numpy as np
import os
from tracker.sort import Sort
from counter.line_counter import LineCounter

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

tracker = Sort()
# 修改: 添加 line_y 参数
counter = LineCounter(line_y=0)  # 初始化时设置 line_y，后续会动态更新

def find_available_camera():
    """Automatically detect available camera devices"""
    print("[INFO] Searching for available cameras...")

    # Try camera indices 0-9
    for i in range(10):
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            print(f"[INFO] Device path exists: {device_path}")
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                ret, frame = temp_cap.read()
                if ret:
                    print(f"[INFO] Found available camera at index: {i}")
                    temp_cap.release()
                    return i
                else:
                    print(f"[WARNING] Camera {i} opened but failed to read frame")
                    temp_cap.release()
            else:
                print(f"[WARNING] Failed to open camera at index: {i}")
        else:
            print(f"[INFO] Device path does not exist: {device_path}")
    
    raise RuntimeError("No camera found")

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


net = cv2.dnn.readNetFromONNX("models/yolov5n_person.onnx")
CAMERA_INDEX = find_available_camera()
cap = cv2.VideoCapture(CAMERA_INDEX)

#降低分辨率减少算力压力
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# 设置窗口大小
cv2.namedWindow("YOLOv5n Person", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5n Person", FRAME_WIDTH, FRAME_HEIGHT)

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 统计线设为画面一半
    LINE_Y = frame.shape[0] // 2
    # 修改: 更新 line_y
    counter.set_line_y(LINE_Y)

    persons = yolo_v5_person_infer(frame, net)
    tracks = tracker.update(persons)
    counter.update(tracks)

    in_count, out_count = counter.get_counts()
    total_count = counter.total_count  # 获取总人数

    # 可视化
    for x1, y1, x2, y2, score in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # 画统计线
    cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (255, 0, 0), 2)

    # 显示统计信息
    cv2.putText(frame, f"Total count: {total_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"IN: {in_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"OUT: {out_count}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow("YOLOv5n Person", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()