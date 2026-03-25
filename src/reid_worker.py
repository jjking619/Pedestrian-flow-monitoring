import threading
import queue
import numpy as np
from reid_extractor import ReIDExtractor

class ReIDWorker(threading.Thread):
    def __init__(self, model_path, max_queue=20):
        super().__init__(daemon=True)
        self.extractor = ReIDExtractor(model_path)
        self.task_queue = queue.Queue(maxsize=max_queue)
        self.feature_dict = {}      # track_id → feature
        self.stop_event = threading.Event()

        dummy = np.zeros((256, 128, 3), dtype=np.uint8)
        self.extractor.extract_feature(dummy, [0, 0, 128, 256])

    def submit(self, track_id, frame, tlwh):
        if self.stop_event.is_set():
            return
        try:
            self.task_queue.put_nowait((track_id, frame.copy(), tlwh))
        except queue.Full:
            pass  

    def get_feature(self, track_id):
        return self.feature_dict.get(track_id)

    def run(self):
        while not self.stop_event.is_set():
            try:
                track_id, frame, tlwh = self.task_queue.get(timeout=0.5)
                feature = self.extractor.extract_feature(frame, tlwh)
                self.feature_dict[track_id] = feature
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ReIDWorker] error: {e}")

    def stop(self):
        self.stop_event.set()