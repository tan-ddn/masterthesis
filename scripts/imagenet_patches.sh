#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x imagenet_patches.sh
/work/scratch/tnguyen/miniconda3/envs/dinov2/bin/python /home/students/tnguyen/masterthesis/segment_patches.py "$@"