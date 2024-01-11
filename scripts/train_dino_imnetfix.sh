#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_dino_imnetfix.sh
/home/students/tnguyen/miniconda3/envs/4p2p_env/bin/python /home/students/tnguyen/masterthesis/dino/eval_linear.py "$@"