#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x imagenet_patches.sh
/home/students/tnguyen/miniconda3/envs/4p2p_env/bin/python /home/students/tnguyen/masterthesis/segment_patches.py "$@"