import numpy as np
from alfred.pose import Person
from alfred.monitor import NightMonitor, Zone


def mk(cx, top, bot, horiz=False):
    k = np.zeros((17, 2)); c = np.ones(17)
    if horiz:
        k[:, 0] = np.linspace(cx - 150, cx + 150, 17); k[:, 1] = bot - 20
        k[5:7, 0] = cx - 120; k[11:13, 0] = cx + 60
    else:
        k[:, 1] = np.linspace(top, bot, 17); k[:, 0] = cx
        k[5, 0], k[6, 0] = cx - 20, cx + 20; k[11, 0], k[12, 0] = cx - 15, cx + 15
        k[5:7, 1] = top + 50; k[11:13, 1] = top + 150
    box = np.array([k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()])
    return Person(k, c, box, "rgb")


def run(drop_s, floor_s=25):
    m = NightMonitor([Zone("bed", 0, 0.35, 0.3, 1)])
    t, dt = 0.0, 0.1
    for _ in range(30):
        m.update(t, mk(500, 100, 400), 640, 480); t += dt
    n = int(drop_s / dt)
    for i in range(n):
        m.update(t, mk(500, 100 + 200 * (i + 1) / n, 400), 640, 480); t += dt
    for _ in range(int(floor_s / dt)):
        m.update(t, mk(500, 380, 420, True), 640, 480); t += dt
    return [e.kind for e in m.events], m.events


def test_fast_fall_calls_family_immediately():
    kinds, _ = run(0.6)
    assert "fall_stage1" in kinds and "fall_stage2" in kinds


def test_slow_lie_down_only_checks_in_then_escalates():
    kinds, _ = run(4.0)
    assert "fall_stage1" in kinds and "fall_stage2" not in kinds and "fall_stage3" in kinds


def test_getting_up_recovers():
    m = NightMonitor([Zone("bed", 0, 0.35, 0.3, 1)])
    t = 0.0
    for _ in range(30):
        m.update(t, mk(500, 100, 400), 640, 480); t += 0.1
    for _ in range(20):
        m.update(t, mk(500, 380, 420, True), 640, 480); t += 0.1
    for _ in range(20):
        m.update(t, mk(500, 100, 400), 640, 480); t += 0.1
    kinds = [e.kind for e in m.events]
    assert "recovered" in kinds and "fall_stage3" not in kinds
