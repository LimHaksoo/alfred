# Alfred

**Always with your parents, even while you sleep.**

Alfred watches over older adults who are alone at night. It reads only body joints from a thermal camera (no faces), notices risky night-time movement, and responds before and after a fall: a voice prompt and a path light first, then a call to the family within seconds.

Built at c0mpiled-17 Seoul (2026-11-09), theme: YC RFS "AI for the Aging Population".

## Results (built today)

| | Before | After |
|---|---|---|
| Thermal pose: people lying down found (OpenThermalPose2 test, 2,151 images rotated 90°) | 31% | **97%** |
| Thermal pose: joint accuracy on people lying down (pose mAP50) | 0.00 | **0.92** |
| Fall decision from joints only: real falls → call family now (UR Fall Detection, 30 falls) | 18 / 30 | **26 / 30** |
| Fall decision from joints only: daily activities → call family now (40 clips) | 17 / 40 | **4 / 40** |

- [`train/`](train/): thermal pose fine-tune with rotation augmentation (11,931 thermal images).
- [`eval/`](eval/): head-drop rule that sees only 17 joint coordinates per frame, decided online (past frames only).

## How it works

1. **Pose**: a thermal YOLO11 pose model finds 17 joints per person. We fine-tuned OpenThermalPose2 with rotation augmentation so it also finds people lying on the floor ([`train/`](train/)); if the thermal model finds nobody, the general RGB pose model runs on the same frame.
2. **Joints only**: every step after pose sees only the 17 joint coordinates per frame, decided online (past frames only).
3. **Fall response in 3 stages** (head-drop rule, [`eval/headdrop.py`](eval/headdrop.py)). Head height = (ankle level while standing − head y) / standing body height, both learned online.
   - head drops by ≥ 0.65 standing heights within 1 s and reaches floor level (< 0.35) → **call family now**
   - at floor level for ≥ 0.5 s without such a fast drop (slow slump) → **voice check-in** ("Are you okay?")
   - still at floor level after 20 s → **call family**
4. **Night context**: bed and bathroom zones separate lying in bed from lying on the floor, and track sit-to-stand time, sway after standing and time on the floor.

The live demo loop (`run.py`, `alfred/monitor.py`) still uses the earlier rule (body extent + descent speed, the "Before" column above); the head-drop rule evaluated above is in `eval/` and is being moved into the live loop next.

## Run

```bash
pip install -r requirements.txt
python run.py --src 0 --show                # webcam
python run.py --src clip.mp4 --out outputs  # video file
```

Family alerts use Telegram: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in your environment.
