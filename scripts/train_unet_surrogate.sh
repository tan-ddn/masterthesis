#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_unet_surrogate.sh
/work/scratch/tnguyen/miniconda3/envs/dinov2/bin/python /home/students/tnguyen/masterthesis/unet/train_surrogate.py "$@"