# Fall decisions from joint coordinates only

Step 1 turns video into joints; every step after that sees only 17 (x, y) joint coordinates per frame.

| Step | Script | Output |
|---|---|---|
| 1. Joints | `extract_joints.py` | `joints/<clip>.npz`: keypoints, confidences, box per frame (YOLO11m-pose) |
| 2. Decision | `headdrop.py` | stages per clip, online (each frame sees only the past) |
| 3. Video | `render_skeleton.py` | black-background skeleton video with the stage and a head-height timeline |

## Rule (head drop)

Head height = (ankle level while standing − head y) / standing body height, both learned online from the joints.

1. **Stage 2, call family now**: the head fell by ≥ 0.65 standing heights within 1 s and is at floor level (< 0.35).
2. **Stage 1, voice check-in**: at floor level for ≥ 0.5 s without such a fast drop.
3. **Stage 3, call family**: still at floor level after 20 s.

The drop threshold 0.65 was chosen by leave-one-out on the clips below; every fold picked 0.64–0.67.

## Results: UR Fall Detection, all 70 clips (30 falls, 40 daily activities)

RGB camera clips, since no public thermal fall video was downloadable today; the rule only sees joints, so
the joint source (thermal or RGB model) can be swapped.

| | Earlier rule (body extent + descent speed) | Head-drop rule |
|---|---|---|
| Falls → call family now | 18 / 30 | **26 / 30** |
| Falls → any alert | 25 / 30 | **29 / 30** |
| Daily activities → call family now | 17 / 40 | **4 / 40** |
| Daily activities → any alert | 26 / 40 | 12 / 40 |

- Crouching to pick something up and sitting down on a chair: no "call family" alarms.
- The 4 daily-activity alarms are all lying down to floor level (adl-30, 33, 38, 39). In a home, lying in bed is
  handled by the bed zone; lying down on the floor quickly is genuinely hard to tell from a fall by joints alone.
- Missed: fall-03 (no alert); fall-08, 12, 20 got the voice check-in only (slower slumps).
- URFD fall clips end right after impact, so stage 3 (20 s on the floor) cannot be tested on them.

Per-clip results: `urfd_headdrop_results.csv`. Scripts expect the data layout under `~/alfred-prep` used on our GPU server.
