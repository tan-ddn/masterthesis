import sys
import torch
import torchvision


PATH = r"/home/students/tnguyen/masterthesis/dinov2_lib/dinov2_vits14_linear_head.pth"

ckpt = torch.load(PATH)
for key, value in ckpt.items():
    print(f"key {key}")
    # if key not in ["optimizer", "scheduler", "iteration"]: