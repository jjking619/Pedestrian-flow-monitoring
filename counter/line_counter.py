class LineCounter:
    def __init__(self, line_y, offset=10):
        """
        line_y: 统计线的 y 坐标
        offset: 容错范围（像素）
        """
        self.line_y = line_y
        self.offset = offset

        self.track_history = {}   # id -> last center y
        self.counted_ids = set()

        self.in_count = 0
        self.out_count = 0

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        """
        for x1, y1, x2, y2, tid in tracks:
            cy = (y1 + y2) // 2

            if tid not in self.track_history:
                self.track_history[tid] = cy
                continue

            last_cy = self.track_history[tid]

            # 已统计的不再重复统计
            if tid in self.counted_ids:
                self.track_history[tid] = cy
                continue

            # 下穿（IN）
            if last_cy < self.line_y - self.offset and cy > self.line_y + self.offset:
                self.in_count += 1
                self.counted_ids.add(tid)

            # 上穿（OUT）
            elif last_cy > self.line_y + self.offset and cy < self.line_y - self.offset:
                self.out_count += 1
                self.counted_ids.add(tid)

            self.track_history[tid] = cy

    def get_counts(self):
        return self.in_count, self.out_count
    
    # 新增方法：设置统计线位置
    def set_line_y(self, line_y):
        """
        设置统计线的 y 坐标
        :param line_y: 新的统计线 y 坐标
        """
        self.line_y = line_y