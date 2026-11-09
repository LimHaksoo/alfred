"""Night monitor: state machine, body signals and the 3-stage fall response.

Fall response (validated on URFD depth features, 2026-11-08):
  1. low near the floor outside the bed for >= 0.5 s  -> voice check-in
  2. final continuous descent faster than 0.35 standing-heights/s -> call family now
  3. not back on their feet within N s (default 20) -> call family
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .pose import Person


@dataclass
class Zone:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float  # normalized 0..1

    def contains(self, p, w: int, h: int) -> bool:
        if p is None:
            return False
        x, y = p[0] / w, p[1] / h
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class Config:
    low_ratio: float = 0.5  # height < half of standing height = "low"
    low_hold_s: float = 0.5
    fast_descent: float = 0.35  # standing-heights per second
    recover_s: float = 20.0
    stand_ratio: float = 0.8
    upright_deg: float = 30.0
    walk_speed: float = 0.25  # standing-heights per second, horizontal
    sway_window_s: float = 10.0
    lost_as_fall_s: float = 2.0


@dataclass
class Event:
    t: float
    kind: str
    detail: dict = field(default_factory=dict)


class NightMonitor:
    def __init__(self, zones: list[Zone], cfg: Config | None = None):
        self.zones = {z.name: z for z in zones}
        self.cfg = cfg or Config()
        self.stand_h: float | None = None  # standing reference height (px)
        self.hist: deque = deque(maxlen=300)  # (t, h_ratio, pelvis_x)
        self.state = "unknown"
        self.state_since = 0.0
        self.events: list[Event] = []
        # signal bookkeeping
        self._rise_start: float | None = None
        self._stood_at: float | None = None
        self._sway_x: list[float] = []
        self._sway_sw: list[float] = []
        # fall bookkeeping
        self._low_since: float | None = None
        self.fall_stage = 0
        self.fall_t: float | None = None
        self._last_seen: float | None = None
        self._last_state_seen = "unknown"

    # ---------- helpers ----------
    def _emit(self, t, kind, **detail):
        ev = Event(t, kind, detail)
        self.events.append(ev)
        return ev

    def _in(self, zone, p, w, h):
        z = self.zones.get(zone)
        return z.contains(p, w, h) if z else False

    def _final_descent_speed(self) -> float:
        """Speed of the last continuous drop in height ratio (standing-heights/s)."""
        if len(self.hist) < 3:
            return 0.0
        pts = list(self.hist)
        h_final = pts[-1][1]
        j = len(pts) - 1
        while j > 0 and abs(pts[j - 1][1] - h_final) < 0.03:  # skip the plateau on the floor
            j -= 1
        t_end, h_end, _ = pts[j]
        i = j
        while i > 0 and pts[i - 1][1] >= pts[i][1] - 0.02:  # walk back the continuous drop
            i -= 1
        while i < j and pts[i + 1][1] >= pts[i][1] - 0.01:  # skip the standing plateau before it
            i += 1
        t0, h0, _ = pts[i]
        return (h0 - h_end) / (t_end - t0) if t_end > t0 and h0 > h_end else 0.0

    # ---------- main ----------
    def update(self, t: float, person: Person | None, w: int, h: int) -> list[Event]:
        n0 = len(self.events)
        cfg = self.cfg

        if person is None:
            if (self._last_seen is not None and self._last_state_seen in ("standing", "walking")
                    and t - self._last_seen > cfg.lost_as_fall_s and self.fall_stage == 0):
                self._start_fall(t, reason="disappeared")
            self._fall_timers(t, recovered=False)
            return self.events[n0:]

        self._last_seen = t
        ext = person.vertical_extent
        angle = person.torso_angle_deg or 0.0
        upright = angle < cfg.upright_deg
        if upright and (self.stand_h is None or ext > self.stand_h * 0.9):
            self.stand_h = ext if self.stand_h is None else 0.9 * self.stand_h + 0.1 * max(ext, self.stand_h)
        ratio = ext / self.stand_h if self.stand_h else 1.0
        if not upright:
            ratio = min(ratio, 0.45)  # a horizontal torso is low regardless of extent
        pel = person.pelvis
        px = float(pel[0]) if pel is not None else float((person.box[0] + person.box[2]) / 2)
        self.hist.append((t, ratio, px))

        in_bed = self._in("bed", pel, w, h)
        in_bath = self._in("bathroom", pel, w, h)
        hspeed = 0.0
        if len(self.hist) >= 5 and self.stand_h:
            t0, _, x0 = self.hist[-5]
            hspeed = abs(px - x0) / self.stand_h / max(t - t0, 1e-3)

        if in_bed and not upright:
            state = "lying"
        elif in_bed:
            state = "sitting"
        elif in_bath:
            state = "bathroom"
        elif ratio < cfg.low_ratio:
            state = "floor"
        elif ratio >= cfg.stand_ratio and upright:
            state = "walking" if hspeed > cfg.walk_speed else "standing"
        else:
            state = "sitting"

        self._signals(t, state, person)
        if state != self.state:
            self._emit(t, "state", frm=self.state, to=state)
            self.state, self.state_since = state, t
        self._last_state_seen = state

        # ---- fall logic ----
        if state == "floor":
            self._low_since = self._low_since or t
            if self.fall_stage == 0 and t - self._low_since >= cfg.low_hold_s:
                self._start_fall(t, reason="low_outside_bed")
        else:
            self._low_since = None
        self._fall_timers(t, recovered=state in ("standing", "walking", "lying", "sitting") and self.fall_stage > 0)
        return self.events[n0:]

    def _start_fall(self, t, reason):
        speed = self._final_descent_speed()
        self.fall_t = t
        self.fall_stage = 1
        self._emit(t, "fall_stage1", reason=reason, descent_speed=round(speed, 3),
                   action="voice_checkin")
        if speed > self.cfg.fast_descent:
            self.fall_stage = 2
            self._emit(t, "fall_stage2", descent_speed=round(speed, 3), action="call_family")

    def _fall_timers(self, t, recovered: bool):
        if self.fall_stage == 0:
            return
        if recovered:
            self._emit(t, "recovered", floor_s=round(t - self.fall_t, 1))
            self.fall_stage, self.fall_t = 0, None
            return
        if self.fall_stage == 1 and t - self.fall_t >= self.cfg.recover_s:
            self.fall_stage = 3
            self._emit(t, "fall_stage3", floor_s=round(t - self.fall_t, 1), action="call_family")

    def _signals(self, t, state, person: Person):
        cfg = self.cfg
        # sit-to-stand time: leaving lying -> first upright standing
        if self.state == "lying" and state != "lying":
            self._rise_start = t
        if state in ("standing", "walking") and self._rise_start is not None:
            self._emit(t, "sit_to_stand", seconds=round(t - self._rise_start, 2))
            self._rise_start = None
            self._stood_at = t
            self._sway_x, self._sway_sw = [], []
        # sway: pelvis lateral std / shoulder width during the first 10 s after standing
        if self._stood_at is not None:
            if t - self._stood_at <= cfg.sway_window_s:
                if person.pelvis is not None and person.shoulder_width:
                    self._sway_x.append(float(person.pelvis[0]))
                    self._sway_sw.append(person.shoulder_width)
            else:
                if len(self._sway_x) >= 5:
                    sway = float(np.std(self._sway_x) / np.median(self._sway_sw))
                    self._emit(t, "post_stand_sway", value=round(sway, 3))
                self._stood_at = None
