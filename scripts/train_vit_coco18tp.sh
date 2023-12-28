#!/bin/bash
cd /home/students/tnguyen/masterthesis/scripts
chmod +x train_vit_coco18tp.sh
/home/students/tnguyen/miniconda3/envs/4p2p_env/bin/python /home/students/tnguyen/masterthesis/model_ViT.py "$@"