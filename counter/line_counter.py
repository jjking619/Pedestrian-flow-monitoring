class LineCounter:
    def __init__(self, line_y=0, offset=5):
        # 保留基本属性，但不再用于进出计数
        self.line_y = line_y
        self.offset = offset
        self.total_count = 0

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        只统计当前画面中的总人数
        """
        self.total_count = len(tracks)

    def get_counts(self):
        # 返回0, 0作为进出计数，只返回总人数
        return 0, 0, self.total_count

    def set_line_y(self, line_y):
        self.line_y = line_y