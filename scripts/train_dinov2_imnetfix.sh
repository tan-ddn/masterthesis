#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_dinov2_imnetfix.sh
/work/scratch/tnguyen/miniconda3/envs/dinov2/bin/python /home/students/tnguyen/masterthesis/dinov2_tan/attn_eval_linear.py "$@"