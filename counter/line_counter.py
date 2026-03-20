class LineCounter:
    def __init__(self, line_y=0, offset=5, max_tracks=1000):
        self.line_y = line_y
        self.offset = offset
        self.max_tracks = max_tracks
        
        # 实时总人数（当前帧中的人数）
        self.total_count = 0
        
        # 进出累计计数（基于计数线穿越）
        self.in_count = 0
        self.out_count = 0
        
        # 累计总人数（基于track_id，所有出现过的不同人员总数）
        self.total_unique_count = 0
        self.seen_track_ids = set()  # 记录所有出现过的track_id
        
        # 跟踪每个track_id的最后位置和穿越状态（用于进出计数）
        self.track_positions = {}  # {track_id: last_y_center}
        self.crossed_tracks = set()  # 已经穿越过计数线的track_id，防止重复计数
        
        # 状态跟踪（用于更精确的穿越判定）
        self.track_states = {}  # {track_id: 'above', 'below', 'crossing_up', 'crossing_down'}

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        更新计数逻辑，包括实时总人数、进出累计人数和累计总人数
        """
        current_track_ids = set()
        current_count = len(tracks)
        
        # 更新实时总人数
        self.total_count = current_count
        
        # 处理每个跟踪目标
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            y_center = (y1 + y2) / 2  # 使用中心点进行穿越判断
            
            current_track_ids.add(track_id)
            
            # 统计累计总人数：记录所有出现过的track_id
            if track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track_id)
                self.total_unique_count += 1
            
            # 初始化新track的状态（用于进出计数）
            if track_id not in self.track_positions:
                self.track_positions[track_id] = y_center
                # 初始状态判断
                if y_center < self.line_y - self.offset:
                    self.track_states[track_id] = 'above'
                elif y_center > self.line_y + self.offset:
                    self.track_states[track_id] = 'below'
                else:
                    self.track_states[track_id] = 'crossing'  # 在计数线附近
                continue
            
            # 获取之前的y坐标和状态
            prev_y = self.track_positions[track_id]
            prev_state = self.track_states.get(track_id, 'unknown')
            
            # 更新当前位置
            self.track_positions[track_id] = y_center
            
            # 如果已经穿越过，跳过计数（防止重复计数）
            if track_id in self.crossed_tracks:
                # 更新状态但不计数
                if y_center < self.line_y - self.offset:
                    self.track_states[track_id] = 'above'
                elif y_center > self.line_y + self.offset:
                    self.track_states[track_id] = 'below'
                continue
            
            # 检测快速移动目标是否跨过了计数线（即使跳过了容差区域）
            crossed_line_directly = False
            if (prev_y <= self.line_y and y_center >= self.line_y) or \
               (prev_y >= self.line_y and y_center <= self.line_y):
                # 确保不是在容差区域内抖动
                if abs(prev_y - self.line_y) > self.offset or abs(y_center - self.line_y) > self.offset:
                    crossed_line_directly = True
            
            # 判断是否发生穿越（原有逻辑）
            crossing_up = (prev_y >= self.line_y - self.offset and 
                          y_center < self.line_y - self.offset and 
                          prev_state != 'above')
            
            crossing_down = (prev_y <= self.line_y + self.offset and 
                            y_center > self.line_y + self.offset and 
                            prev_state != 'below')
            
            # 处理向上穿越（进入区域）
            if crossing_up or (crossed_line_directly and y_center < self.line_y):
                self.in_count += 1
                self.crossed_tracks.add(track_id)
                self.track_states[track_id] = 'above'
                continue
            
            # 处理向下穿越（离开区域）
            if crossing_down or (crossed_line_directly and y_center > self.line_y):
                self.out_count += 1
                self.crossed_tracks.add(track_id)
                self.track_states[track_id] = 'below'
                continue
            
            # 更新状态（未穿越的情况）
            if y_center < self.line_y - self.offset:
                self.track_states[track_id] = 'above'
            elif y_center > self.line_y + self.offset:
                self.track_states[track_id] = 'below'
            else:
                self.track_states[track_id] = 'crossing'
        
        # 清理消失的track（只清理位置和状态，不清理seen_track_ids）
        disappeared_tracks = set(self.track_positions.keys()) - current_track_ids
        for track_id in disappeared_tracks:
            if track_id in self.track_positions:
                del self.track_positions[track_id]
            if track_id in self.track_states:
                del self.track_states[track_id]
            if track_id in self.crossed_tracks:
                self.crossed_tracks.discard(track_id)
        
        # 防止内存无限增长 - 清理最老的记录（只针对活跃跟踪）
        if len(self.track_positions) > self.max_tracks:
            # 获取最老的track_id（最小的ID）
            oldest_track_id = min(self.track_positions.keys())
            if oldest_track_id in self.track_positions:
                del self.track_positions[oldest_track_id]
            if oldest_track_id in self.track_states:
                del self.track_states[oldest_track_id]
            if oldest_track_id in self.crossed_tracks:
                self.crossed_tracks.discard(oldest_track_id)

    def get_counts(self):
        """
        返回进出计数、累计总人数和实时总人数
        Returns: (in_count, out_count, total_unique_count, total_count)
        """
        return self.in_count, self.out_count, self.total_unique_count, self.total_count

    def set_line_y(self, line_y):
        """设置计数线的Y坐标"""
        self.line_y = line_y
        
    def reset_counts(self):
        """重置累计计数（保留实时计数）"""
        self.in_count = 0
        self.out_count = 0
        self.total_unique_count = 0
        self.crossed_tracks.clear()
        self.seen_track_ids.clear()
        self.seen_person_features.clear()