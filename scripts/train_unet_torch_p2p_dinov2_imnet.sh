#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_unet_torch_p2p_dinov2_imnetfix.sh
/work/scratch/tnguyen/miniconda3/envs/dinov2/bin/python /home/students/tnguyen/masterthesis/unet/train_encoder.py "$@"
