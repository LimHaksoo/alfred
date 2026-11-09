"""Pose extraction with a thermal-first, RGB-fallback path.

The thermal model (OpenThermalPose2 YOLO11) is accurate on upright people but
misses many lying ones, so when it finds nobody we re-run the general RGB
model on the same frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# COCO-17 keypoint indices
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK = 0, 5, 6, 11, 12, 15, 16


@dataclass
class Person:
    kpts: np.ndarray  # (17, 2) pixel coords
    conf: np.ndarray  # (17,)
    box: np.ndarray  # (4,) x1 y1 x2 y2
    source: str  # "thermal" | "rgb"

    def pt(self, i: int, min_conf: float = 0.3):
        return self.kpts[i] if self.conf[i] >= min_conf else None

    def mid(self, a: int, b: int, min_conf: float = 0.3):
        pa, pb = self.pt(a, min_conf), self.pt(b, min_conf)
        if pa is not None and pb is not None:
            return (pa + pb) / 2
        return pa if pa is not None else pb

    @property
    def pelvis(self):
        return self.mid(L_HIP, R_HIP)

    @property
    def shoulders(self):
        return self.mid(L_SH, R_SH)

    @property
    def shoulder_width(self) -> float | None:
        a, b = self.pt(L_SH), self.pt(R_SH)
        return float(np.linalg.norm(a - b)) if a is not None and b is not None else None

    @property
    def vertical_extent(self) -> float:
        """Top-to-bottom span of visible keypoints (falls back to the box)."""
        vis = self.kpts[self.conf >= 0.3]
        if len(vis) >= 5:
            return float(vis[:, 1].max() - vis[:, 1].min())
        return float(self.box[3] - self.box[1])

    @property
    def torso_angle_deg(self) -> float | None:
        """0 = upright, 90 = horizontal."""
        s, p = self.shoulders, self.pelvis
        if s is None or p is None:
            w, h = self.box[2] - self.box[0], self.box[3] - self.box[1]
            return 90.0 if w > h * 1.2 else 0.0
        d = s - p
        return float(np.degrees(np.arctan2(abs(d[0]), abs(d[1]) + 1e-6)))


class PoseEstimator:
    def __init__(self, rgb_weights: str = "yolo11s-pose.pt", thermal_weights: str | None = None,
                 conf: float = 0.35, device: str | None = None):
        self.rgb = YOLO(rgb_weights)
        self.thermal = YOLO(thermal_weights) if thermal_weights and Path(thermal_weights).exists() else None
        self.conf = conf
        self.device = device

    def _run(self, model, frame, source: str) -> list[Person]:
        r = model.predict(frame, conf=self.conf, verbose=False, device=self.device)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return []
        k = r.keypoints.xy.cpu().numpy()
        c = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(k.shape[:2])
        b = r.boxes.xyxy.cpu().numpy()
        return [Person(k[i], c[i], b[i], source) for i in range(len(b))]

    def __call__(self, frame) -> Person | None:
        """Return the single most prominent person (one resident per room)."""
        people = self._run(self.thermal, frame, "thermal") if self.thermal else []
        if not people:
            people = self._run(self.rgb, frame, "rgb")
        if not people:
            return None
        return max(people, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
