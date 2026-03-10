import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        """
        bbox: [x1, y1, x2, y2]
        """
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])

        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = np.array(bbox).reshape((4,1))

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 1

    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(np.array(bbox).reshape((4,1)))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape(-1)