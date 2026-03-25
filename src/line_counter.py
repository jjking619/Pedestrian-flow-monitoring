import numpy as np
from collections import defaultdict

class LineCounter:
    """
    基于虚拟线的行人流量统计器
    支持水平线和垂直线两种统计模式
    """
    def __init__(self, line_position=None, direction='horizontal', max_tracks=1000):
        self.max_tracks = max_tracks
        
        # 虚拟线位置和方向
        self.line_pos = line_position
        self.direction = direction  # 'horizontal' 或 'vertical'
        
        # 进出计数
        self.in_count = 0
        self.out_count = 0
        self.total_count = 0
        
        # 实时当前帧人数
        self.current_count = 0
        
        # 轨迹历史记录 {track_id: [(x, y), ...]}
        self.track_history = defaultdict(list)
        
        # 已统计过的轨迹ID集合，避免重复计数
        self.counted_tracks = set()
        
        # 记录所有见过的track_ids（用于兼容原有total_count逻辑）
        self.seen_track_ids = set()

    def update(self, tracks, frame_shape=None):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        frame_shape: (height, width) - 图像尺寸，用于自动设置虚拟线位置
        Update counting logic with virtual line-based flow counting
        """
        self.current_count = len(tracks)
        
        # 自动设置虚拟线位置（如果未指定）
        if self.line_pos is None and frame_shape is not None:
            if self.direction == 'horizontal':
                self.line_pos = frame_shape[0] // 2  # 图像中间高度
            else:
                self.line_pos = frame_shape[1] // 2  # 图像中间宽度
        elif self.line_pos is None:
            # 如果没有frame_shape信息，使用默认值
            self.line_pos = 360
        
        # 更新seen_track_ids（保持原有total_count逻辑兼容性）
        for track in tracks:
            if len(track) >= 5:
                track_id = track[4]
                if track_id not in self.seen_track_ids:
                    self.seen_track_ids.add(track_id)
        
        # 如果没有虚拟线位置，只做简单计数（兼容原有功能）
        if self.line_pos is None:
            self.total_count = len(self.seen_track_ids)
            return
        
        # 基于虚拟线的进出统计
        current_tracks = {}
        
        # 处理当前帧的跟踪结果
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = track[:5]
                # 计算边界框中心点
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                center = (center_x, center_y)
                
                current_tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'center': center
                }
                
                # 更新轨迹历史
                self.track_history[track_id].append(center)
                if len(self.track_history[track_id]) > 30:  # 限制历史长度
                    self.track_history[track_id].pop(0)
        
        # 检查轨迹是否穿过统计线
        for track_id, track_data in current_tracks.items():
            if track_id in self.counted_tracks:
                continue
                
            history = self.track_history[track_id]
            if len(history) >= 2:
                prev_center = history[-2]
                curr_center = history[-1]
                
                if self._cross_line(prev_center, curr_center):
                    direction = self._get_direction(prev_center, curr_center)
                    if direction == 'in':
                        self.in_count += 1
                    else:
                        self.out_count += 1
                    self.total_count = self.in_count + self.out_count
                    self.counted_tracks.add(track_id)

    def _cross_line(self, p1, p2):
        """判断两点连线是否穿过统计线"""
        if self.direction == 'horizontal':
            # 水平线：检查y坐标是否跨过line_pos
            return (p1[1] < self.line_pos and p2[1] >= self.line_pos) or \
                   (p1[1] > self.line_pos and p2[1] <= self.line_pos)
        else:
            # 垂直线：检查x坐标是否跨过line_pos
            return (p1[0] < self.line_pos and p2[0] >= self.line_pos) or \
                   (p1[0] > self.line_pos and p2[0] <= self.line_pos)
    
    def _get_direction(self, p1, p2):
        """判断移动方向"""
        if self.direction == 'horizontal':
            # 水平线：向下移动为进入，向上移动为离开
            return 'in' if p2[1] > p1[1] else 'out'
        else:
            # 垂直线：向右移动为进入，向左移动为离开
            return 'in' if p2[0] > p1[0] else 'out'

    def get_counts(self):
        """
        Return cumulative total count and real-time total count
        Returns: (current_count, total_count, in_count, out_count)
        """
        return self.current_count, self.total_count, self.in_count, self.out_count
    
    def reset(self):
        """重置计数器"""
        self.in_count = 0
        self.out_count = 0
        self.total_count = 0
        self.current_count = 0
        self.track_history.clear()
        self.counted_tracks.clear()
        self.seen_track_ids.clear()
    
    def get_line_info(self):
        """获取统计线信息"""
        return {
            'position': self.line_pos,
            'direction': self.direction
        }