class LineCounter:
    def __init__(self, line_y, offset=5):
        """
        line_y: 统计线的 y 坐标
        offset: 容错范围（像素）
        """
        self.line_y = line_y
        self.offset = offset

        self.track_history = {}   # id -> last center y
        
        # 实时计数（每帧重置）
        self.current_in_count = 0  # 当前帧新增的进入人数
        self.current_out_count = 0  # 当前帧新增的离开人数
        self.total_count = 0       # 当前总人数
        
        # 已经穿越过统计线的跟踪ID集合，避免重复计数
        self.crossed_tracks = set()

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        基于行人穿越统计线的方向进行实时进出统计
        """
        self.current_in_count = 0
        self.current_out_count = 0
        self.total_count = len(tracks)
        
        
        for x1, y1, x2, y2, tid in tracks:
            cy = (y1 + y2) // 2

            if tid not in self.track_history:
                self.track_history[tid] = cy
                print(f"  [NEW] ID={tid}, current_cy={cy}, line_y={self.line_y}")
                continue

            last_cy = self.track_history[tid]
            
            # 检查是否已经穿越过统计线，避免重复计数
            if tid not in self.crossed_tracks:
                # 下穿（IN）- 从线上方到线下方
                if last_cy < self.line_y - self.offset and cy > self.line_y + self.offset:
                    self.current_in_count += 1
                    self.crossed_tracks.add(tid)
                    print(f"  [✓ IN] ID={tid}, last_cy={last_cy} -> cy={cy}, crossed line_y={self.line_y}")
                # 上穿（OUT）- 从线下方到线上方  
                elif last_cy > self.line_y + self.offset and cy < self.line_y - self.offset:
                    self.current_out_count += 1
                    self.crossed_tracks.add(tid)
                    print(f"  [✓ OUT] ID={tid}, last_cy={last_cy} -> cy={cy}, crossed line_y={self.line_y}")
            else:
                print(f"  [X] ID={tid}, last_cy={last_cy} -> cy={cy}, crossed line_y={self.line_y}")
                # 更新历史位置
            self.track_history[tid] = cy
        
        if self.current_in_count > 0 or self.current_out_count > 0:
            print(f"  [RESULT] IN={self.current_in_count}, OUT={self.current_out_count}, TOTAL={self.total_count}")

    def get_counts(self):
        """返回当前帧的实时进出统计：(in_count, out_count, total_count)"""
        return self.current_in_count, self.current_out_count, self.total_count
    
    
    def set_line_y(self, line_y):
        """
        设置统计线的 y 坐标
        :param line_y: 新的统计线 y 坐标
        """
        self.line_y = line_y
        