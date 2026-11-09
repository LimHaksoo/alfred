# Thermal pose fine-tune for lying and fallen bodies

The OpenThermalPose2 YOLO11l-pose model is accurate on upright people but loses people who are lying down,
which is exactly the moment Alfred must not miss. These scripts retrain it today with rotation augmentation.

Run on a GPU server (we used two RTX A6000):

```bash
CUDA_VISIBLE_DEVICES=0,1 python train/train_thermal_rot.py 60 l 0,1   # epochs, model size, GPUs
CUDA_VISIBLE_DEVICES=0 python train/eval_rotation.py runs/thermal_rot_l_ft2g/weights/best.pt
```

`eval_rotation.py` scores a model on the OpenThermalPose2 test split (2,151 images) as-is and rotated 90 degrees
each way, with boxes and keypoints rotated along with the image.

## Results so far (OpenThermalPose2 test, 2,151 images)

| Model | Upright joint mAP50-95 | Rotated 90° recall | Rotated 90° joint mAP50 |
|---|---|---|---|
| OpenThermalPose2 YOLO11l (original) | 0.888 | 0.31 | 0.00 |
| General YOLO11m (RGB COCO) | 0.787 | 0.78–0.80 | 0.19 |
| **Ours, fine-tuned (60 epochs)** | 0.788 | **0.97** | **0.91–0.92** |

Rotated joint mAP50-95 for ours is 0.71–0.72 (the original model: 0.00). Upright accuracy dropped from 0.888 to 0.788,
about the level of the general RGB model, in exchange for tracking people at any angle. Training ran 60 epochs
(~1 h on two RTX A6000); best validation joint mAP50-95 0.837. Settings that mattered: AdamW at lr 0.0005
(the default SGD lr 0.01 wiped out the pretrained model in one epoch), rotation up to 90 degrees.
