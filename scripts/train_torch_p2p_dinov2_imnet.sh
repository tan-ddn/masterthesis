#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_torch_p2p_dinov2_imnetfix.sh
/work/scratch/tnguyen/miniconda3/envs/dinov2/bin/python /home/students/tnguyen/masterthesis/p2p/torch_p2p_dinov2_linear_classifier.py "$@"
