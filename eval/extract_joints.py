"""Step 1 of the joints-only pipeline: turn each UR Fall Detection clip into joint coordinates and nothing else.
cam0 MP4s are depth|RGB side by side; the RGB half is upscaled x2 (640x480) and run through YOLO11m-pose.
The most prominent person per frame is kept, like alfred/pose.py. Everything after this step reads only
~/alfred-prep/outputs/joints/<clip>.npz: kpts (T,17,2) px, conf (T,17), box (T,4), all NaN when nobody is found.
"""
import glob, os
import cv2, numpy as np
from ultralytics import YOLO

BASE = os.path.expanduser('~/alfred-prep')
OUT = f'{BASE}/outputs/joints'
os.makedirs(OUT, exist_ok=True)
model = YOLO(f'{BASE}/models/yolo11m-pose.pt')

for path in sorted(glob.glob(f'{BASE}/data/urfd_mp4/*-cam0.mp4')):
    clip = os.path.basename(path)[:-9]
    if os.path.exists(f'{OUT}/{clip}.npz'):
        continue
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    K, C, B = [], [], []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f[:, f.shape[1] // 2:], None, fx=2, fy=2)
        r = model.predict(f, device=0, conf=0.35, verbose=False)[0]
        if len(r.boxes):
            b = r.boxes.xyxy.cpu().numpy()
            i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
            K.append(r.keypoints.xy.cpu().numpy()[i]); C.append(r.keypoints.conf.cpu().numpy()[i]); B.append(b[i])
        else:
            K.append(np.full((17, 2), np.nan)); C.append(np.zeros(17)); B.append(np.full(4, np.nan))
    np.savez_compressed(f'{OUT}/{clip}.npz', kpts=np.array(K, np.float32), conf=np.array(C, np.float32),
                        box=np.array(B, np.float32), fps=fps, size=np.array([640, 480]))
    print(f'JOINTS {clip} frames {len(K)} with person {int(np.isfinite(np.array(B)[:, 0]).sum())}', flush=True)
print('JOINTS DONE', flush=True)
