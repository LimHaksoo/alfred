"""Compare pose models on the OpenThermalPose2 test split as-is and rotated 90 degrees both ways.

Rotated copies keep correct labels (boxes and keypoints rotated with the image), so pose mAP is
measured on sideways bodies, not just detection rate.

  CUDA_VISIBLE_DEVICES=3 nice -n 19 python eval_rotation.py <weights.pt> [<weights.pt> ...]
"""
import glob
import os
import sys

import cv2

BASE = os.path.expanduser('~/alfred-prep')
TEST = f'{BASE}/data/openthermalpose2/otp2_dataset/test'
ROT_ROOT = f'{BASE}/data/otp2_test_rotated'
FLIP_IDX = '[0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]'


def rot_point(x, y, how):
    # normalized coords; cw: (x, y) -> (1 - y, x); ccw: (x, y) -> (y, 1 - x)
    return (1 - y, x) if how == 'cw' else (y, 1 - x)


def build(how):
    d = f'{ROT_ROOT}/{how}'
    if os.path.isdir(f'{d}/labels') and len(os.listdir(f'{d}/labels')) > 0:
        return d
    os.makedirs(f'{d}/images', exist_ok=True)
    os.makedirs(f'{d}/labels', exist_ok=True)
    code = cv2.ROTATE_90_CLOCKWISE if how == 'cw' else cv2.ROTATE_90_COUNTERCLOCKWISE
    for img in sorted(glob.glob(f'{TEST}/images/*')):
        stem = os.path.splitext(os.path.basename(img))[0]
        lab = f'{TEST}/labels/{stem}.txt'
        if not os.path.exists(lab):
            continue
        cv2.imwrite(f'{d}/images/{stem}.png', cv2.rotate(cv2.imread(img), code))
        out = []
        for line in open(lab):
            v = line.split()
            if len(v) < 5:
                continue
            cx, cy, w, h = map(float, v[1:5])
            ncx, ncy = rot_point(cx, cy, how)
            parts = [v[0], f'{ncx:.6f}', f'{ncy:.6f}', f'{h:.6f}', f'{w:.6f}']
            kp = v[5:]
            for i in range(0, len(kp), 3):
                x, y, vis = float(kp[i]), float(kp[i + 1]), kp[i + 2]
                if float(vis) == 0:
                    parts += ['0', '0', vis]
                else:
                    nx, ny = rot_point(x, y, how)
                    parts += [f'{nx:.6f}', f'{ny:.6f}', vis]
            out.append(' '.join(parts))
        open(f'{d}/labels/{stem}.txt', 'w').write('\n'.join(out) + '\n')
    return d


def yaml_for(path, name):
    y = f'{ROT_ROOT}/{name}.yaml'
    os.makedirs(ROT_ROOT, exist_ok=True)
    open(y, 'w').write(f"path: {path}\ntrain: images\nval: images\nkpt_shape: [17, 3]\n"
                       f"flip_idx: {FLIP_IDX}\nnames:\n  0: person\n")
    return y


if __name__ == '__main__':
    from ultralytics import YOLO
    sets = {'upright': yaml_for(TEST, 'upright'),
            'rot90_cw': yaml_for(build('cw'), 'rot90_cw'),
            'rot90_ccw': yaml_for(build('ccw'), 'rot90_ccw')}
    for w in sys.argv[1:]:
        for name, y in sets.items():
            m = YOLO(w).val(data=y, split='val', device=0, verbose=False, plots=False, batch=16)
            print(f'EVAL {os.path.basename(os.path.dirname(os.path.dirname(w))) or w:28s} {name:10s} '
                  f'box mAP50 {m.box.map50:.3f}  recall {m.box.mr:.3f}  pose mAP50 {m.pose.map50:.3f}  '
                  f'pose mAP50-95 {m.pose.map:.3f}', flush=True)
    print('EVAL DONE', flush=True)
