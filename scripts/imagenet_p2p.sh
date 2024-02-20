#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x imagenet_p2p.sh
/work/scratch/tnguyen/miniconda3/envs/dinov2/bin/python /home/students/tnguyen/masterthesis/dinov2_tan/pulse2percept_layer.py "$@"