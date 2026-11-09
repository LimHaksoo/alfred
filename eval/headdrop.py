"""Online, joints-only fall monitor (head-drop rule), evaluated on all URFD clips.

Input per frame: 17 joint (x, y) and confidences. Nothing else.
  head height = (ankle level while standing - head y) / standing body height, both references learned online
  stage 2 (call family now): head fell >= DROP standing heights within the last 1 s and is at floor level (< LOW)
  stage 1 (voice check-in):  head at floor level for >= 0.5 s without such a fast drop (slow slump, lying down)
  stage 3 (call family):     still at floor level after RECOVER_S
  recovered:                 head back above 0.7
Thresholds: DROP 0.65 was picked by leave-one-out on the 70 URFD clips (every fold chose 0.64-0.67).
"""
import glob, os
from collections import deque
import numpy as np

DROP, LOW, LOW_HOLD_S, RECOVER_S, UP = 0.65, 0.35, 0.5, 20.0, 0.7


class HeadDropMonitor:
    def __init__(self, fps):
        self.fps = fps
        self.ext_hist, self.ank_hist = [], []
        self.hh = deque(maxlen=int(round(fps)) + 1)  # last 1 s of smoothed head height
        self.raw = deque(maxlen=5)
        self.stage, self.low_since, self.fall_t, self.events = 0, None, None, []
        self.head_h = np.nan

    def update(self, t, kpts, conf):
        ok = conf >= 0.3
        if ok.sum() < 5:
            self.hh.append(np.nan)
            self._timers(t, None)
            return self.head_h
        ext = float(np.ptp(kpts[ok][:, 1])) if ok.sum() >= 8 else np.nan
        ank = float(kpts[15:, 1][ok[15:]].mean()) if ok[15:].any() else np.nan
        self.ext_hist.append(ext); self.ank_hist.append(ank)
        e = np.array(self.ext_hist); a = np.array(self.ank_hist)
        if not np.isfinite(e).any():
            return self.head_h
        tall = np.isfinite(e) & (e >= np.nanpercentile(e, 70))  # standing frames seen so far
        stand = np.nanmedian(e[tall])
        floor = np.nanmedian(a[tall & np.isfinite(a)]) if (tall & np.isfinite(a)).any() else np.nanmax(a)
        head = float(kpts[:5, 1][ok[:5]].mean()) if ok[:5].any() else np.nan
        self.raw.append((floor - head) / stand if stand > 0 else np.nan)
        r = np.array(self.raw)
        self.head_h = float(np.nanmean(r)) if np.isfinite(r).any() else np.nan
        self.hh.append(self.head_h)
        self._timers(t, self.head_h)
        return self.head_h

    def _emit(self, t, kind, **d):
        self.events.append((round(t, 2), kind, d))

    def _timers(self, t, h):
        if h is None or not np.isfinite(h):
            return
        if self.stage and h > UP:
            self._emit(t, 'recovered', down_s=round(t - self.fall_t, 1))
            self.stage, self.fall_t, self.low_since = 0, None, None
            return
        if h < LOW:
            self.low_since = self.low_since if self.low_since is not None else t
            win = np.array(self.hh)
            drop = float(np.nanmax(win) - h) if np.isfinite(win).any() else 0.0
            if self.stage < 2 and drop >= DROP:
                if self.stage == 0:
                    self.fall_t = t
                    self._emit(t, 'fall_stage1', drop=round(drop, 2))
                self.stage = 2
                self._emit(t, 'fall_stage2', drop=round(drop, 2))
            elif self.stage == 0 and t - self.low_since >= LOW_HOLD_S:
                self.stage, self.fall_t = 1, t
                self._emit(t, 'fall_stage1', drop=round(drop, 2))
            if self.stage == 1 and t - self.fall_t >= RECOVER_S:
                self.stage = 3
                self._emit(t, 'fall_stage3', down_s=round(t - self.fall_t, 1))
        else:
            self.low_since = None


def run_clip(path):
    z = np.load(path)
    fps = float(z['fps'])
    m = HeadDropMonitor(fps)
    trace = []
    for i in range(len(z['kpts'])):
        trace.append((i / fps, m.update(i / fps, z['kpts'][i], z['conf'][i]), m.stage))
    return m, trace, z


if __name__ == '__main__':
    BASE = os.path.expanduser('~/alfred-prep')
    res = {}
    for p in sorted(glob.glob(f'{BASE}/outputs/joints/*.npz')):
        clip = os.path.basename(p)[:-4]
        m, _, _ = run_clip(p)
        kinds = [k for _, k, _ in m.events]
        res[clip] = ('fall_stage2' in kinds, 'fall_stage1' in kinds)
        print(f"CLIP {clip:7s} {'FALL' if clip.startswith('fall') else 'adl '} call_now={'Y' if res[clip][0] else '-'} "
              f"any={'Y' if res[clip][1] else '-'} {m.events}")
    for truth in ['fall', 'adl']:
        g = {c: v for c, v in res.items() if c.startswith(truth)}
        print(f'SUM {truth:4s} n={len(g)} call_family_now={sum(v[0] for v in g.values())} '
              f'any_alert={sum(v[1] for v in g.values())}')
