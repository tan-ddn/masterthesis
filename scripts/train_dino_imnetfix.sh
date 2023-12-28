#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_dino_imnetfix.sh
/home/students/tnguyen/miniconda3/envs/4p2p_env/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=25680 /home/students/tnguyen/masterthesis/dino/eval_linear.py "$@"