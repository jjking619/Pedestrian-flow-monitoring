import cv2
import numpy as np

class ReIDExtractor:
    """
    基于 OSNet 的行人重识别特征提取器
    使用 ONNX 模型为检测到的行人提取外观特征向量
    """
    def __init__(self, model_path="models/osnet_x0_25_market1501.onnx"):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = (256, 128)  # OSNet 标准输入尺寸
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def extract_feature(self, frame, tlwh):
        """
        为单个检测到的行人提取特征
        
        Args:
            frame: 输入图像帧
            tlwh: 边界框坐标 [top, left, width, height]
            
        Returns:
            feature: 归一化的特征向量 (512 维)，如果提取失败则返回 None
        """
        try:
            x, y, w, h = map(int, tlwh)
            
            # 确保边界框在图像范围内
            h_frame, w_frame = frame.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            if w <= 0 or h <= 0:
                return None
            
            # 裁剪行人区域
            person_img = frame[y:y+h, x:x+w]
            
            if person_img.size == 0:
                return None
            
            # 调整到模型输入尺寸
            person_img = cv2.resize(person_img, self.input_size)
            
            # 归一化 (RGB 顺序，因为 OSNet 是在 RGB 图像上训练的)
            person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            person_img = person_img.astype(np.float32) / 255.0
            person_img = (person_img - self.mean) / self.std
            
            # 转换为 NCHW 格式
            blob = cv2.dnn.blobFromImage(
                person_img,
                scalefactor=1.0,
                size=self.input_size,
                swapRB=False,  # 已经手动转换了 RGB
                crop=False
            )
            
            # 前向推理
            self.net.setInput(blob)
            feature = self.net.forward()
            
            # L2 归一化特征向量
            feature = feature.flatten()
            norm = np.linalg.norm(feature)
            if norm > 0:
                feature = feature / norm
            
            return feature
            
        except Exception as e:
            return None
    
    def batch_extract(self, frame, detections):
        """
        批量提取多个行人的特征
        Args:
            frame: 输入图像帧
            detections: 检测结果列表，每个元素包含 tlwh 信息
            
        Returns:
            features: 特征向量列表，每个元素对应一个检测
        """
        features = []
        for det in detections:
            # 支持不同的检测对象格式
            if hasattr(det, 'tlwh'):
                tlwh = det.tlwh
            elif isinstance(det, np.ndarray):
                tlwh = det[:4]
            else:
                tlwh = det
            
            feature = self.extract_feature(frame, tlwh)
            features.append(feature)
        
        return features
