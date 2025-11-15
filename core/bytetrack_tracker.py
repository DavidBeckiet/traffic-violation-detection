import numpy as np
from scipy.optimize import linear_sum_assignment

class STrack:
    _id_count = 0

    def __init__(self, tlwh, score, cls):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.score = score
        self.cls = cls
        self.track_id = STrack.next_id()
        self.is_activated = True

    @staticmethod
    def next_id():
        STrack._id_count += 1
        return STrack._id_count

    def to_tlbr(self):
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)


class BYTETracker:
    def __init__(self, track_thresh=0.5, match_thresh=0.8):
        """
        track_thresh: ngưỡng confidence để tạo track
        match_thresh: ngưỡng IoU để match
        """
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.tracked_tracks = []

    def update(self, detections):
        """
        detections: list of (x1, y1, x2, y2, conf, cls)
        """
        dets = []
        for (x1, y1, x2, y2, conf, cls) in detections:
            w, h = x2 - x1, y2 - y1
            if conf >= self.track_thresh:
                dets.append([x1, y1, w, h, conf, cls])

        dets = np.array(dets, dtype=np.float32)
        new_tracks = []

        # ====================
        # Match với track cũ
        # ====================
        if len(self.tracked_tracks) > 0 and len(dets) > 0:
            iou_matrix = self.compute_iou(self.tracked_tracks, dets)

            row_idx, col_idx = linear_sum_assignment(-iou_matrix)

            matched = []
            for r, c in zip(row_idx, col_idx):
                if iou_matrix[r, c] >= self.match_thresh:
                    track = self.tracked_tracks[r]
                    x, y, w, h = dets[c][:4]
                    track.tlwh = np.array([x, y, w, h])
                    matched.append(r)

            # Track giữ lại
            updated_tracks = [t for i, t in enumerate(self.tracked_tracks) if i in matched]

        else:
            updated_tracks = []

        # ====================
        # Thêm track mới
        # ====================
        used_det = set()
        for _, c in zip(row_idx, col_idx):
            used_det.add(c)

        for i, det in enumerate(dets):
            if i not in used_det:
                x, y, w, h, conf, cls = det
                new_tracks.append(STrack([x, y, w, h], conf, cls))

        self.tracked_tracks = updated_tracks + new_tracks

        return self.tracked_tracks

    @staticmethod
    def compute_iou(tracks, detections):
        """ Tính IoU giữa các track và detection """
        ious = np.zeros((len(tracks), len(detections)), dtype=np.float32)

        for i, t in enumerate(tracks):
            tbox = t.to_tlbr()
            for j, d in enumerate(detections):
                dbox = np.array([d[0], d[1], d[0] + d[2], d[1] + d[3]])
                ious[i, j] = BYTETracker.iou(tbox, dbox)

        return ious

    @staticmethod
    def iou(bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])

        w = max(0., xx2 - xx1)
        h = max(0., yy2 - yy1)
        inter = w * h

        area_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_test + area_gt - inter

        return inter / union if union > 0 else 0
