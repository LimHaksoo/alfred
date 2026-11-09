"""Run Alfred on a video file or webcam.

  python run.py --src 0                       # webcam
  python run.py --src clip.mp4 --out outputs/ # file, writes annotated video + events.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from alfred.alerts import VOICE, Notifier, say
from alfred.monitor import Config, NightMonitor, Zone
from alfred.pose import PoseEstimator

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLORS = {"floor": (0, 0, 255), "bathroom": (255, 160, 0), "lying": (180, 180, 180)}


def load_zones(path: str | None) -> list[Zone]:
    if not path:
        return [Zone("bed", 0.0, 0.35, 0.3, 1.0)]
    return [Zone(**z) for z in json.loads(Path(path).read_text())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="0")
    ap.add_argument("--zones", default=None, help="JSON list of {name,x1,y1,x2,y2} (normalized)")
    ap.add_argument("--rgb", default="yolo11s-pose.pt")
    ap.add_argument("--thermal", default=None, help="OpenThermalPose2 weights (optional)")
    ap.add_argument("--recover-s", type=float, default=20.0)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-alerts", action="store_true")
    a = ap.parse_args()

    src = int(a.src) if a.src.isdigit() else a.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {a.src!r} (webcam: check macOS camera permission)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    live = isinstance(src, int)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = "webcam" if live else Path(a.src).stem
    writer = None

    pose = PoseEstimator(a.rgb, a.thermal)
    zones = load_zones(a.zones)
    mon = NightMonitor(zones, Config(recover_s=a.recover_s))
    notify = Notifier()
    evf = open(out / f"{name}_events.jsonl", "w")
    t_start, frame_i, banner = time.time(), 0, ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t = (time.time() - t_start) if live else frame_i / fps
        frame_i += 1
        person = pose(frame)
        for ev in mon.update(t, person, w, h):
            evf.write(json.dumps({"t": round(ev.t, 2), "kind": ev.kind, **ev.detail}) + "\n")
            evf.flush()
            if ev.kind != "state":
                banner = f"{ev.kind} {ev.detail}"
            if a.no_alerts:
                continue
            detected = time.time()
            if ev.kind == "fall_stage1":
                say(VOICE["voice_checkin"])
            elif ev.kind in ("fall_stage2", "fall_stage3"):
                notify.family(f"Alfred: possible fall ({ev.kind}, {ev.detail}). Please check now.", detected)
            elif ev.kind == "state" and ev.detail.get("frm") == "lying":
                say(VOICE["bed_exit"])

        # ---- draw ----
        for z in zones:
            cv2.rectangle(frame, (int(z.x1 * w), int(z.y1 * h)), (int(z.x2 * w), int(z.y2 * h)), (90, 90, 90), 1)
            cv2.putText(frame, z.name, (int(z.x1 * w) + 4, int(z.y1 * h) + 16), 0, 0.5, (150, 150, 150), 1)
        if person is not None:
            col = COLORS.get(mon.state, (0, 220, 120))
            for i, j in SKELETON:
                if person.conf[i] > 0.3 and person.conf[j] > 0.3:
                    cv2.line(frame, tuple(map(int, person.kpts[i])), tuple(map(int, person.kpts[j])), col, 2)
        cv2.putText(frame, f"{mon.state}  stage={mon.fall_stage}", (10, 28), 0, 0.8, (255, 255, 255), 2)
        if mon.state != "floor" and mon.fall_stage == 0 and "state" not in banner:
            cv2.putText(frame, banner[:70], (10, h - 14), 0, 0.5, (0, 255, 255), 1)
        if mon.fall_stage:
            cv2.putText(frame, "PATH LIGHT ON", (w - 190, 28), 0, 0.7, (0, 255, 255), 2)

        if writer is None:
            writer = cv2.VideoWriter(str(out / f"{name}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(frame)
        if a.show:
            cv2.imshow("Alfred", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    evf.close()
    (out / f"{name}_alerts.json").write_text(json.dumps(notify.log, indent=2))
    print(f"frames={frame_i} events={len(mon.events)} -> {out}")


if __name__ == "__main__":
    main()
