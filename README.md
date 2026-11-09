# Alfred

**Always there. Even when you're asleep.**

Alfred watches over older adults who are alone at night. It reads only body joints from a thermal camera (no faces), notices risky night-time movement, and responds before and after a fall: a voice prompt and a path light first, then a call to the family within seconds.

Built at c0mpiled-17 Seoul (2026-11-09), theme: YC RFS "AI for the Aging Population".

## How it works

1. **Pose**: YOLO11 pose. A thermal model (OpenThermalPose2) runs first; if it finds nobody (it misses many lying people), the general RGB model runs on the same frame.
2. **State**: lying → sitting → standing → walking → bathroom → floor, using bed/bathroom zones.
3. **Body signals**: sit-to-stand time, pelvis sway after standing (÷ shoulder width), time on the floor.
4. **Fall response in 3 stages**
   - low near the floor outside the bed → voice check-in
   - final descent faster than 0.35 standing-heights/s → call family now
   - not back up within N s (default 20) → call family

## Run

```bash
pip install -r requirements.txt
python run.py --src 0 --show                # webcam
python run.py --src clip.mp4 --out outputs  # video file
```

Family alerts use Telegram: set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in your environment.
