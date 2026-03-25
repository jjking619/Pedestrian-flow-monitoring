import numpy as np
from collections import OrderedDict
import time

class LineCounter:
    def __init__(self, max_tracks=1000, reid_threshold=0.6, feature_buffer_size=50):
        self.max_tracks = max_tracks
        self.reid_threshold = reid_threshold  # ReID相似度阈值
        self.feature_buffer_size = feature_buffer_size
        
        # 实时计数（当前帧中的人数）
        self.current_count = 0
        
        # 累计唯一人数计数（基于ReID特征去重）
        self.total_count = 0
        
        # 基于track_id的简单统计（备用）
        self.seen_track_ids = set()
        
        # ReID特征库：存储已见过人员的特征向量
        self.known_features = OrderedDict()  # {feature_hash: (last_seen_time, feature_vector)}
        self.feature_history = {}  # {track_id: [features]}
        
        # 时间戳用于特征老化清理
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 60  # 每60秒清理一次过期特征

    def _hash_feature(self, feature):
        """生成特征向量的哈希值用于快速查找"""
        if feature is None:
            return None
        return hash(tuple(np.round(feature, decimals=6)))

    def _is_similar_to_known(self, new_feature):
        """检查新特征是否与已知特征相似"""
        if new_feature is None or len(self.known_features) == 0:
            return False
            
        # 计算与所有已知特征的余弦相似度
        for feat_hash, (_, known_feat) in self.known_features.items():
            if known_feat is not None:
                similarity = np.dot(new_feature, known_feat)
                if similarity >= self.reid_threshold:
                    return True
        return False

    def _add_known_feature(self, feature):
        """添加新的已知特征到特征库"""
        if feature is None:
            return
            
        feat_hash = self._hash_feature(feature)
        current_time = time.time()
        self.known_features[feat_hash] = (current_time, feature)
        
        # 限制特征库大小，移除最老的记录
        if len(self.known_features) > self.max_tracks:
            self.known_features.popitem(last=False)

    def _cleanup_old_features(self):
        """清理过期的特征记录"""
        current_time = time.time()
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return
            
        # 移除超过5分钟未见的特征
        cutoff_time = current_time - 300  # 5分钟
        to_remove = []
        for feat_hash, (last_seen, _) in self.known_features.items():
            if last_seen < cutoff_time:
                to_remove.append(feat_hash)
                
        for feat_hash in to_remove:
            del self.known_features[feat_hash]
            
        self.last_cleanup_time = current_time

    def update(self, tracks, persons, track_features=None):
        """
        更新计数逻辑
        
        Args:
            tracks: [[x1,y1,x2,y2,id], ...]
            persons: 检测到的人员列表
            track_features: 对应每个track的ReID特征列表
        """
        # 更新实时计数
        self.current_count = len(persons)
        
        # 清理过期特征
        self._cleanup_old_features()
        
        if track_features is None:
            track_features = [None] * len(tracks)
            
        # 处理每个track
        for i, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track
            feature = track_features[i] if i < len(track_features) else None
            
            # 基于track_id的简单统计（备用方案）
            if track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track_id)
            
            # 基于ReID特征的去重统计（主要方案）
            if feature is not None:
                if not self._is_similar_to_known(feature):
                    # 发现新人员
                    self._add_known_feature(feature)
                    self.total_count += 1
            else:
                # 如果没有特征，回退到track_id方案
                pass

    def get_counts(self):
        return self.current_count, self.total_count
