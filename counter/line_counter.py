import numpy as np

class LineCounter:
    def __init__(self, max_tracks=1000):
        self.max_tracks = max_tracks
        
        # 实时总人数（当前帧中的人数）
        self.total_count = 0
        
        # 累计总人数（基于track_id，所有出现过的不同人员总数）
        self.total_unique_count = 0
        self.seen_track_ids = set()  # 记录所有出现过的track_id

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        更新计数逻辑，只保留实时总人数和累计总人数
        """
        current_count = len(tracks)
        
        # 更新实时总人数
        self.total_count = current_count
        
        # 处理每个跟踪目标，统计累计总人数
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            
            # 统计累计总人数：记录所有出现过的track_id
            if track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track_id)
                self.total_unique_count += 1

    def get_counts(self):
        """
        返回累计总人数和实时总人数
        Returns: (total_unique_count, total_count)
        """
        return self.total_unique_count, self.total_count

    def reset_counts(self):
        """重置累计计数"""
        self.total_unique_count = 0
        self.seen_track_ids.clear()