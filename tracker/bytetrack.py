
import numpy as np
import numba
from numba import njit, prange
from scipy.optimize import linear_sum_assignment
from tracker.reid_extractor import ReIDExtractor

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class BaseTrack(object):
    _count = 0
    track_id = 0
    is_activated = False
    state = TrackState.New
    history = {}
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

class KalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = np.dot(covariance, self._update_mat.T)
        kalman_gain = np.dot(kalman_gain, np.linalg.inv(projected_cov))
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def multi_predict(self, mean, covariance):
        if len(mean) == 0:
            return mean, covariance
        
        std_pos = np.array([
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ])
        std_vel = np.array([
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ])
        sqr = np.square(np.r_[std_pos, std_vel]).T
        motion_cov = np.array([np.diag(s) for s in sqr])
        
        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feature=None):
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        # 新增特征相关
        self.features = []          # 保存历史特征（可限制长度）
        self.curr_feature = None    # 当前帧的特征
        if feature is not None:
            self.curr_feature = feature
            self.features.append(feature)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, feature=None):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        if feature is not None:
            self.curr_feature = feature
            self.features.append(feature)
            if len(self.features) > 50:
                self.features.pop(0)

    def get_feature(self):
    # 返回最近的特征（或平均特征）
        if len(self.features) > 0:
            return self.features[-1]
        else:
            return None
    def update(self, new_track, frame_id, feature=None):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        # 更新特征
        if feature is not None:
            self.curr_feature = feature
            self.features.append(feature)
            # 可选：限制特征列表长度，防止内存膨胀
            if len(self.features) > 50:
                self.features.pop(0)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def compute_feature_distance(tracks, detections):
    """
    计算轨迹和检测之间的外观特征距离矩阵
    
    Args:
        tracks: 轨迹对象列表 (STrack 实例)
        detections: 检测对象列表 (STrack 实例)
        
    Returns:
        feat_dist: 特征距离矩阵 (M, N)，使用余弦距离
    """
    m, n = len(tracks), len(detections)
    if m == 0 or n == 0:
        return np.zeros((m, n))
    
    feat_dist = np.zeros((m, n), dtype=np.float32)
    
    for i in range(m):
        track_feat = tracks[i].curr_feature
        if track_feat is None:
            feat_dist[i, :] = 1.0  # 如果轨迹无特征，设为默认距离
            continue
        
        for j in range(n):
            det_feat = detections[j].curr_feature
            if det_feat is None:
                feat_dist[i, j] = 1.0
                continue
            
            # 余弦相似度 = dot(a,b) / (|a|*|b|)，因为已归一化，直接点积
            sim = np.dot(track_feat, det_feat)
            # 距离 = 1 - 相似度 (余弦距离范围 0~2，但归一化后相似度接近 1，距离接近 0~0.几)
            feat_dist[i, j] = 1.0 - sim
    
    return feat_dist

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    resa, resb = [], []
    for track in stracksa:
        resa.append(track)
    for track in stracksb:
        resb.append(track)
    return resa, resb

def ious(atlbrs, btlbrs):
    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)))
    
    atlbrs = np.array(atlbrs)
    btlbrs = np.array(btlbrs)
    
    inter_x1 = np.maximum(atlbrs[:, None, 0], btlbrs[:, 0])
    inter_y1 = np.maximum(atlbrs[:, None, 1], btlbrs[:, 1])
    inter_x2 = np.minimum(atlbrs[:, None, 2], btlbrs[:, 2])
    inter_y2 = np.minimum(atlbrs[:, None, 3], btlbrs[:, 3])
    
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    union_area = area_a[:, None] + area_b - inter_area
    
    iou = inter_area / np.maximum(union_area, 1e-6)
    return iou

# @njit(parallel=True, fastmath=True,nogil=True,nopython=True)
def compute_iou_matrix(atlbrs, btlbrs):
    """
    计算两组边界框（tlbr格式）的 IOU 矩阵。
    输入 atlbrs: (M, 4) 数组
    输入 btlbrs: (N, 4) 数组
    返回 IOU 距离矩阵 (1 - IOU)，形状 (M, N)
    """
    m = atlbrs.shape[0]
    n = btlbrs.shape[0]
    iou_matrix = np.empty((m, n), dtype=np.float32)

    for i in prange(m):  # 并行外层循环
        a_x1, a_y1, a_x2, a_y2 = atlbrs[i]
        a_area = (a_x2 - a_x1) * (a_y2 - a_y1)
        if a_area <= 0:
            a_area = 1e-6

        for j in range(n):
            b_x1, b_y1, b_x2, b_y2 = btlbrs[j]
            b_area = (b_x2 - b_x1) * (b_y2 - b_y1)
            if b_area <= 0:
                b_area = 1e-6

            # 交集
            inter_x1 = max(a_x1, b_x1)
            inter_y1 = max(a_y1, b_y1)
            inter_x2 = min(a_x2, b_x2)
            inter_y2 = min(a_y2, b_y2)
            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            # IOU
            union_area = a_area + b_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0.0
            iou_matrix[i, j] = 1.0 - iou  # 距离

    return iou_matrix

def iou_distance(atracks, btracks):
    # 原有代码：将输入统一为二维数组 atlbrs, btlbrs
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = np.array([track.tlbr for track in atracks])
        btlbrs = np.array([track.tlbr for track in btracks])

    if len(atlbrs) == 0 or len(btlbrs) == 0:
        return np.zeros((len(atlbrs), len(btlbrs)))

    # 确保形状为 (n,4)
    if atlbrs.shape[1] < 4 or btlbrs.shape[1] < 4:
        atlbrs = atlbrs[:, :4]
        btlbrs = btlbrs[:, :4]

    if atlbrs.ndim != 2 or btlbrs.ndim != 2:
        atlbrs = atlbrs.reshape(-1, 4)
        btlbrs = btlbrs.reshape(-1, 4)

    # 调用 Numba 加速函数
    return compute_iou_matrix(atlbrs.astype(np.float32), btlbrs.astype(np.float32))

# @njit(parallel=True, fastmath=True,nogil=True,nopython=True)
def fuse_score_matrix(cost_matrix, scores):
    """
    cost_matrix: (M, N) IOU 距离矩阵
    scores: (N,) 检测得分数组
    返回融合后的代价矩阵
    """
    m, n = cost_matrix.shape
    fuse_cost = np.empty((m, n), dtype=cost_matrix.dtype)

    for i in prange(m):
        for j in range(n):
            iou_sim = 1.0 - cost_matrix[i, j]
            fuse_sim = iou_sim * scores[j]
            fuse_cost[i, j] = 1.0 - fuse_sim
    return fuse_cost

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    scores = np.array([det.score for det in detections], dtype=np.float32)
    return fuse_score_matrix(cost_matrix, scores)

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    unmatched_rows = []
    unmatched_cols = []
    for r in range(cost_matrix.shape[0]):
        if r not in row_ind:
            unmatched_rows.append(r)
    for c in range(cost_matrix.shape[1]):
        if c not in col_ind:
            unmatched_cols.append(c)

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > thresh:
            unmatched_rows.append(r)
            unmatched_cols.append(c)
        else:
            matches.append((r, c))

    return matches, unmatched_rows, unmatched_cols

def fuse_iou_feat_cost(iou_cost, feat_cost, iou_weight=0.5, feat_weight=0.5):
    """
    融合 IOU 距离和外观特征距离
    
    Args:
        iou_cost: (M,N) IOU 距离矩阵（0~1）
        feat_cost: (M,N) 特征距离矩阵（0~2，通常 0~1）
        iou_weight: IOU 距离权重
        feat_weight: 特征距离权重
        
    Returns:
        融合后的代价矩阵
    """
    return iou_weight * iou_cost + feat_weight * feat_cost


class Args:
    def __init__(self, track_thresh=0.5, high_thresh=0.5, low_thresh=0.1, match_thresh=0.8, track_buffer=30):
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer

class BYTETracker(object):
    def __init__(self, track_thresh=0.5, high_thresh=0.5, low_thresh=0.1, match_thresh=0.8, 
                 track_buffer=30, frame_rate=30, use_reid=False, 
                 reid_model_path="models/osnet_x0_25_market1501.onnx",
                 iou_weight=0.6, feat_weight=0.4):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = Args(track_thresh, high_thresh, low_thresh, match_thresh, track_buffer)
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        # ReID 相关参数
        self.use_reid = use_reid
        if use_reid:
            self.reid_extractor = ReIDExtractor(reid_model_path)
        else:
            self.reid_extractor = None
        
        # 融合权重
        self.iou_weight = iou_weight
        self.feat_weight = feat_weight

    def update(self, output_results, frame=None):
        """
        更新跟踪器
        
        Args:
            output_results: 检测结果，格式为 [x1, y1, x2, y2, score] 或包含 tlwh 属性的对象
            frame: 图像帧，当 use_reid=True 时必需，用于提取 ReID 特征
            
        Returns:
            跟踪结果列表 [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 处理检测结果
        if isinstance(output_results, list):
            if len(output_results) == 0:
                dets = np.empty((0, 4))
                scores = np.empty(0)
            else:
                output_results = np.array(output_results)
                scores = output_results[:, 4]
                dets = output_results[:, :4]
        else:
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                dets = output_results[:, :4]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                dets = output_results[:, :4]

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        
        dets_second = dets[inds_second]
        dets_high = dets[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # 如果需要 ReID 特征，先提取特征
        det_features_high = None
        det_features_second = None
        
        if self.use_reid and frame is not None and self.reid_extractor is not None:
            # 提取高分检测的 ReID 特征
            if len(dets_high) > 0:
                det_features_high = []
                for tlbr in dets_high:
                    tlwh = STrack.tlbr_to_tlwh(tlbr)
                    feat = self.reid_extractor.extract_feature(frame, tlwh)
                    det_features_high.append(feat)
            
            # 提取低分检测的 ReID 特征
            if len(dets_second) > 0:
                det_features_second = []
                for tlbr in dets_second:
                    tlwh = STrack.tlbr_to_tlwh(tlbr)
                    feat = self.reid_extractor.extract_feature(frame, tlwh)
                    det_features_second.append(feat)
        
        # 创建高分检测的 STrack 对象（带特征）
        if len(dets_high) > 0:
            detections = []
            for i, (tlbr, s) in enumerate(zip(dets_high, scores_keep)):
                feat = det_features_high[i] if det_features_high is not None else None
                detections.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, feature=feat))
        else:
            detections = []

        # 分离已激活和未激活的轨迹
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 第一次关联：高分检测框
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        
        # 计算 IOU 距离
        dists = iou_distance(strack_pool, detections)
        
        # 如果使用 ReID，融合外观特征距离
        if self.use_reid and len(strack_pool) > 0 and len(detections) > 0:
            feat_dists = compute_feature_distance(strack_pool, detections)
            dists = fuse_iou_feat_cost(dists, feat_dists, 
                                      iou_weight=self.iou_weight, 
                                      feat_weight=self.feat_weight)
        
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.args.match_thresh)
        
        for itracked, idet in matches:
            if itracked >= len(strack_pool) or idet >= len(detections):
                continue  # 索引保护
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二次关联：低分检测框
        if len(dets_second) > 0:
            detections_second = []
            for i, (tlbr, s) in enumerate(zip(dets_second, scores_second)):
                feat = det_features_second[i] if det_features_second is not None else None
                detections_second.append(STrack(STrack.tlbr_to_tlwh(tlbr), s, feature=feat))
        else:
            detections_second = []

        # 构建用于第二次匹配的轨迹池（只包含状态为 Tracked 的轨迹）
        r_tracked_stracks = []
        for i in u_track:
            if i < len(strack_pool) and strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])

        if len(detections_second) > 0 and len(r_tracked_stracks) > 0:
            # 计算 IOU 距离
            dists = iou_distance(r_tracked_stracks, detections_second)
            
            # 如果使用 ReID，融合外观特征距离
            if self.use_reid:
                feat_dists = compute_feature_distance(r_tracked_stracks, detections_second)
                dists = fuse_iou_feat_cost(dists, feat_dists,
                                          iou_weight=self.iou_weight,
                                          feat_weight=self.feat_weight)
            
            matches, u_track_second, u_detection_second = linear_assignment(dists, thresh=0.5)
            
            for itracked, idet in matches:
                if itracked >= len(r_tracked_stracks) or idet >= len(detections_second):
                    continue
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            
            for it in u_track_second:
                if it >= len(r_tracked_stracks):
                    continue
                track = r_tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            # 如果没有低分检测或轨迹，则所有未匹配的高分轨迹标记为 lost
            for it in u_track:
                if it >= len(strack_pool):
                    continue
                track = strack_pool[it]
                if track.state == TrackState.Tracked and not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        # 处理未确认的轨迹（使用第一次匹配后剩余的高分检测）
        if len(u_detection) > 0:
            # 注意：这里重新构建 detections 为未匹配的高分检测
            detections_unmatched = [detections[i] for i in u_detection if i < len(detections)]
        else:
            detections_unmatched = []

        if len(detections_unmatched) > 0 and len(unconfirmed) > 0:
            dists = iou_distance(unconfirmed, detections_unmatched)
            
            # 如果使用 ReID，融合外观特征距离
            if self.use_reid:
                feat_dists = compute_feature_distance(unconfirmed, detections_unmatched)
                dists = fuse_iou_feat_cost(dists, feat_dists,
                                          iou_weight=self.iou_weight,
                                          feat_weight=self.feat_weight)
            
            dists = fuse_score(dists, detections_unmatched)
            matches, u_unconfirmed, u_detection_final = linear_assignment(dists, thresh=0.7)
            
            for itracked, idet in matches:
                if itracked >= len(unconfirmed) or idet >= len(detections_unmatched):
                    continue
                unconfirmed[itracked].update(detections_unmatched[idet], self.frame_id)
                activated_stracks.append(unconfirmed[itracked])
            
            for it in u_unconfirmed:
                if it >= len(unconfirmed):
                    continue
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
        else:
            u_detection_final = list(range(len(detections_unmatched))) if len(detections_unmatched) > 0 else []
            for it in range(len(unconfirmed)):
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        # 初始化新的轨迹（使用未匹配的高分检测）
        for inew in u_detection_final:
            if inew >= len(detections_unmatched):
                continue
            track = detections_unmatched[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 更新状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 返回跟踪结果
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        result = []
        for track in output_stracks:
            tlbr = track.tlbr
            result.append([int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3]), track.track_id])
        
        return result