"""A video drawn from joint coordinates alone (black background, no camera pixels), with the head-drop monitor's
stage and a head-height timeline. Usage: python render_skeleton.py <out.mp4> <clip> [<clip> ...]
"""
import os, shutil, subprocess, sys
import cv2, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # headdrop.py next to this file
from headdrop import DROP, LOW, run_clip  # noqa: E402

BASE = os.path.expanduser('~/alfred-prep')
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6), (0, 1), (0, 2), (1, 3), (2, 4)]
W, H, TL = 640, 480, 120
GREEN, AMBER, RED = (120, 255, 120), (0, 210, 255), (70, 70, 255)
STAGE = {0: ('', GREEN), 1: ('STAGE 1  voice check-in: "Are you okay?"', AMBER),
         2: ('STAGE 2  fast fall to the floor: calling family now', RED),
         3: ('STAGE 3  still down: calling family', RED)}

out, clips = sys.argv[1], sys.argv[2:]
raw = out.replace('.mp4', '_raw.mp4')
wr = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H + TL))
for clip in clips:
    m, trace, z = run_clip(f'{BASE}/outputs/joints/{clip}.npz')
    kpts, conf, fps = z['kpts'], z['conf'], float(z['fps'])
    T = len(kpts) / fps
    ev = [(t, k) for t, k, _ in m.events if k.startswith('fall_stage')]
    img = None
    for i, (t, hh, stage) in enumerate(trace):
        img = np.zeros((H + TL, W, 3), np.uint8)
        for gy in range(60, H, 60):
            cv2.line(img, (0, gy), (W, gy), (26, 26, 26), 1)
        col = STAGE[stage][1]
        if (conf[i] >= 0.3).sum() >= 5:
            for a, b in SKELETON:
                if conf[i, a] >= 0.3 and conf[i, b] >= 0.3:
                    cv2.line(img, tuple(map(int, kpts[i, a])), tuple(map(int, kpts[i, b])), col, 3, cv2.LINE_AA)
            for j in range(17):
                if conf[i, j] >= 0.3:
                    cv2.circle(img, tuple(map(int, kpts[i, j])), 4, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.putText(img, 'no joints this frame', (W // 2 - 105, H // 2), 0, 0.65, (110, 110, 110), 1, cv2.LINE_AA)
        cv2.putText(img, f'{clip}   t={t:4.1f}s   input: 17 joints (x, y) per frame', (10, 24), 0, 0.52, (190, 190, 190), 1, cv2.LINE_AA)
        hs = f'{hh:.2f}' if hh == hh else '-'
        cv2.putText(img, f'head height: {hs} x standing height', (10, 52), 0, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        if STAGE[stage][0]:
            cv2.putText(img, STAGE[stage][0], (10, 84), 0, 0.66, col, 2, cv2.LINE_AA)
        # timeline strip
        cv2.rectangle(img, (0, H), (W, H + TL), (16, 16, 16), -1)
        cv2.putText(img, 'head height over time', (8, H + 16), 0, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        def xy(tt, r):
            return int(12 + (W - 24) * tt / T), int(H + TL - 10 - (TL - 30) * (min(max(r, -0.2), 1.2) + 0.2) / 1.4)
        ly = xy(0, LOW)[1]
        cv2.line(img, (12, ly), (W - 12, ly), (0, 130, 190), 1)
        cv2.putText(img, f'floor level ({LOW})', (W - 120, ly - 4), 0, 0.38, (0, 130, 190), 1, cv2.LINE_AA)
        pts = [(xy(tt, r), s) for tt, r, s in trace[:i + 1] if r == r]
        for (a, _), (b, s) in zip(pts, pts[1:]):
            cv2.line(img, a, b, STAGE[s][1], 2, cv2.LINE_AA)
        for et, k in ev:
            if et <= t:
                x = xy(et, 0)[0]
                cv2.line(img, (x, H + 22), (x, H + TL - 6), AMBER if k == 'fall_stage1' else RED, 1)
        wr.write(img)
    final = max(s for _, _, s in trace)
    verdict = {0: 'no alert', 1: 'voice check-in', 2: 'family called (fast fall)', 3: 'family called (still down)'}[final]
    cv2.rectangle(img, (0, H // 2 - 34), (W, H // 2 + 22), (0, 0, 0), -1)
    cv2.putText(img, f'{clip}: {verdict}', (20, H // 2 + 4), 0, 0.85, STAGE[final][1] if final else GREEN, 2, cv2.LINE_AA)
    for _ in range(int(fps * 1.5)):
        wr.write(img)
    print(f'RENDER {clip} frames {len(trace)} final_stage {final} events {m.events}', flush=True)
wr.release()
subprocess.run([shutil.which('ffmpeg'), '-y', '-loglevel', 'error', '-i', raw, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-crf', '24', '-movflags', '+faststart', out], check=True)
os.remove(raw)
print('VIDEO', out, os.path.getsize(out), flush=True)
