#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_deit_imnetfix.sh
/home/students/tnguyen/miniconda3/envs/4p2p_env/bin/python -m torch.distributed.run --nproc_per_node=4 /home/students/tnguyen/masterthesis/deit/main.py "$@"
# /home/students/tnguyen/miniconda3/envs/4p2p_env/bin/python -m torch.distributed.launch --nproc_per_node=4 --use_env --master_port=22202 /home/students/tnguyen/masterthesis/deit/main.py "$@"