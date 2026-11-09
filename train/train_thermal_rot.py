"""Fine-tune the OpenThermalPose2 YOLO11l-pose model so it keeps tracking people who are lying or fallen.

Baseline weakness (11/8 rehearsal): rotating upright thermal images by 90 degrees dropped person
detection from 100% to 46.7%. Fix: continue training on the full OpenThermalPose2 train split with
strong rotation augmentation (+-90 degrees) so sideways and horizontal bodies look familiar.

Run on cherry, GPU 3 only:
  CUDA_VISIBLE_DEVICES=3 nice -n 19 python train_thermal_rot.py [epochs] [model_size]
"""
import os
import sys

from ultralytics import YOLO

BASE = os.path.expanduser('~/alfred-prep')
DATA_ROOT = f'{BASE}/data/openthermalpose2/otp2_dataset'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 60
SIZE = sys.argv[2] if len(sys.argv) > 2 else 'l'
# GPUs as seen after CUDA_VISIBLE_DEVICES, e.g. "0,1" for two-GPU DDP. Batch is split across them.
DEVICE = sys.argv[3] if len(sys.argv) > 3 else '0'
N_GPU = len(DEVICE.split(','))

data_yaml = f'{BASE}/otp2_train.yaml'
with open(data_yaml, 'w') as f:
    f.write(
        f"path: {DATA_ROOT}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "kpt_shape: [17, 3]\n"
        "flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
        "names:\n  0: person\n"
    )

model = YOLO(f'{BASE}/models/openthermalpose2/otp2_checkpoints/yolo11{SIZE}-pose.pt')
model.train(
    data=data_yaml,
    epochs=EPOCHS,
    imgsz=640,
    batch=16 * N_GPU,
    device=DEVICE if N_GPU > 1 else int(DEVICE),
    workers=8,
    degrees=90.0,      # the point of this run: bodies at any angle up to sideways
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=10,
    patience=20,
    # Fine-tune, don't retrain: the 'auto' optimizer picked SGD lr 0.01 and val pose mAP50-95
    # fell from ~0.89 to 0.03 in epoch 1. A small AdamW rate keeps the pretrained weights.
    optimizer='AdamW',
    lr0=0.0005,
    warmup_epochs=1,
    cos_lr=True,
    save_period=5,     # usable checkpoints even if we stop early for the demo
    project=f'{BASE}/runs',
    name=f'thermal_rot_{SIZE}_ft{N_GPU}g',
    exist_ok=True,
    plots=True,
)
print('TRAIN DONE', flush=True)
