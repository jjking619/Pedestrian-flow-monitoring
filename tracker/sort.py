import numpy as np
from scipy.optimize import linear_sum_assignment
from tracker.kalman import KalmanBoxTracker


def iou(bb1, bb2):
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    area1 = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    area2 = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])
    return inter / (area1 + area2 - inter + 1e-6)


class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_thresh=0.3):
        self.trackers = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh

    def update(self, detections):
        """
        detections: [[x1,y1,x2,y2,score], ...]
        return: [[x1,y1,x2,y2,id], ...]
        """
        trks = []
        for t in self.trackers:
            trks.append(t.predict())

        matched, unmatched_dets, unmatched_trks = self.associate(
            detections, trks
        )

        for d, t in matched:
            self.trackers[t].update(detections[d][:4])

        for d in unmatched_dets:
            self.trackers.append(
                KalmanBoxTracker(detections[d][:4])
            )

        results = []
        alive = []
        for t in self.trackers:
            if t.time_since_update < 1 and t.hits >= self.min_hits:
                x1,y1,x2,y2 = map(int, t.kf.x[:4])
                results.append([x1,y1,x2,y2,t.id])
            if t.time_since_update < self.max_age:
                alive.append(t)

        self.trackers = alive
        return results

    def associate(self, detections, trackers):
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        iou_mat = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_mat[d, t] = iou(det[:4], trk)

        row, col = linear_sum_assignment(-iou_mat)

        matched = []
        unmatched_dets = []
        unmatched_trks = []

        for d in range(len(detections)):
            if d not in row:
                unmatched_dets.append(d)

        for t in range(len(trackers)):
            if t not in col:
                unmatched_trks.append(t)

        for r, c in zip(row, col):
            if iou_mat[r, c] < self.iou_thresh:
                unmatched_dets.append(r)
                unmatched_trks.append(c)
            else:
                matched.append((r, c))

        return matched, unmatched_dets, unmatched_trks
    
    