import numpy as np

class LineCounter:
    def __init__(self, max_tracks=1000):
        self.max_tracks = max_tracks
        
        # Real-time total count (number of people in current frame)
        self.total_count = 0
        
        # Cumulative total count (total number of unique persons based on track_id)
        self.total_unique_count = 0
        self.seen_track_ids = set()  # Record all seen track_ids

    def update(self, tracks):
        """
        tracks: [[x1,y1,x2,y2,id], ...]
        Update counting logic, keeping only real-time and cumulative counts
        """
        current_count = len(tracks)
        
        self.total_count = current_count
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            
            if track_id not in self.seen_track_ids:
                self.seen_track_ids.add(track_id)
                self.total_unique_count += 1

    def get_counts(self):
        """
        Return cumulative total count and real-time total count
        Returns: (total_unique_count, total_count)
        """
        return self.total_unique_count, self.total_count

    def reset_counts(self):
        """Reset cumulative counts"""
        self.total_unique_count = 0
        self.seen_track_ids.clear()