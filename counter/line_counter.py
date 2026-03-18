class LineCounter:
    def __init__(self, line_y, offset=5):
        self.line_y = line_y
        self.offset = offset

        # id -> last_center_y
        self.last_y = {}

        # 记录每个ID最后的穿越方向，用于防止同一方向重复计数
        self.last_cross_direction = {}  # id -> 'IN' or 'OUT' or None

        self.in_count = 0
        self.out_count = 0
        self.total_count = 0

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        """
        self.in_count = 0
        self.out_count = 0
        self.total_count = len(tracks)

        active_ids = set()
        completion_threshold = self.offset  # 完成穿越后继续显示的距离阈值

        for x1, y1, x2, y2, tid in tracks:
            cy = (y1 + y2) // 2
            active_ids.add(tid)

            if tid not in self.last_y:
                self.last_y[tid] = cy
                self.last_cross_direction[tid] = None
                continue

            last_cy = self.last_y[tid]
            last_direction = self.last_cross_direction[tid]

            # 调试信息
            print(f"DEBUG: Track ID {tid} - last_cy: {last_cy}, cy: {cy}, line_y: {self.line_y}, offset: {self.offset}, last_dir: {last_direction}")
            
            # 检查是否开始新的穿越（完全穿越判定）
            current_direction = None
            
            # IN方向：从线上方完全穿越到线下方
            if (last_cy <= self.line_y - self.offset and 
                cy >= self.line_y + self.offset and 
                last_direction != 'IN'):
                # 使用边界框底部坐标确保完全穿越
                if y2 > self.line_y + self.offset:
                    self.last_cross_direction[tid] = 'IN'
                    current_direction = 'IN'
                    print(f"DEBUG: Track ID {tid} STARTED IN CROSS!")
            
            # OUT方向：从线下方完全穿越到线上方  
            elif (last_cy >= self.line_y + self.offset and 
                  cy <= self.line_y - self.offset and 
                  last_direction != 'OUT'):
                # 使用边界框顶部坐标确保完全穿越
                if y1 < self.line_y - self.offset:
                    self.last_cross_direction[tid] = 'OUT'
                    current_direction = 'OUT'
                    print(f"DEBUG: Track ID {tid} STARTED OUT CROSS!")

            # 判断是否仍在显示区域内（用于实时计数显示）
            if self.last_cross_direction[tid] == 'IN':
                # IN计数持续显示：目标顶部仍在影响区域内
                if y1 <= self.line_y + completion_threshold:
                    self.in_count += 1
                    print(f"DEBUG: Track ID {tid} showing IN count (y1={y1} <= {self.line_y + completion_threshold})")
                else:
                    # 目标已远离，重置方向以便下次穿越
                    self.last_cross_direction[tid] = None
                    print(f"DEBUG: Track ID {tid} IN completed, reset direction")
                    
            elif self.last_cross_direction[tid] == 'OUT':
                # OUT计数持续显示：目标底部仍在影响区域内
                if y2 >= self.line_y - completion_threshold:
                    self.out_count += 1
                    print(f"DEBUG: Track ID {tid} showing OUT count (y2={y2} >= {self.line_y - completion_threshold})")
                else:
                    # 目标已远离，重置方向以便下次穿越
                    self.last_cross_direction[tid] = None
                    print(f"DEBUG: Track ID {tid} OUT completed, reset direction")

            self.last_y[tid] = cy

        # 清理消失目标
        for tid in list(self.last_y.keys()):
            if tid not in active_ids:
                print(f"DEBUG: Track ID {tid} removed from tracking")
                self.last_y.pop(tid, None)
                self.last_cross_direction.pop(tid, None)

    def get_counts(self):
        return self.in_count, self.out_count, self.total_count

    def set_line_y(self, line_y):
        self.line_y = line_y