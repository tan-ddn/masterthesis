import argparse
import logging
import os
import glob
import cv2
import numpy as np
import random
import sys
import json
import pypatchify
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tv_transforms
from torchvision.utils import save_image
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
from random import shuffle


sys.path.append("/home/students/tnguyen/masterthesis")

from p2p.torch_p2p_dinov2_masked_model import *
from unet.unet_model import UNet


class ModelWithMaskedLastFeature(nn.Module):
    def __init__(self, feature_model, top_attention, n_last_blocks, autocast_ctx, args):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.top_attention = top_attention
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
        self.args = args
        self.p2p_patch_size = args.patch_size
        
        if self.args.torch_p2p:
            p2pmodel, implant = build_p2pmodel_and_implant(self.p2p_patch_size, axlambda=args.axlambda, rho=args.rho, range_limit=args.xyrange, xystep=args.xystep)
            # self.phis = get_patient_params(p2pmodel, args.device)
            # self.decoder = AxonMapSpatialModule(p2pmodel, implant, amp_cutoff=True).to(args.device)
            self.decoder = AxonMapSpatialModifiedModule(torch.tensor([[args.rho]]), torch.tensor([[args.axlambda]]), p2pmodel, implant, amp_cutoff=True, chunk_length=self.args.chunk_length).to(args.device)
            for p in self.decoder.parameters():
                p.requires_grad = False
            self.decoder.eval()
            self.percept_norm_for_dino = pth_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                # print(f'top_attention {self.feature_model.top_attention}')
                if self.args.torch_p2p:
                    percepts = image2percept(images, self.p2p_patch_size, self.decoder, self.args.chunk_length)
                    # print(f"percepts shape, max, and min {percepts.shape, percepts.max(), percepts.min()}")
                    features = self.feature_model.get_intermediate_layers_with_masked_feature(
                        self.percept_norm_for_dino(percepts), self.n_last_blocks,
                    )
                    del percepts
                else:
                    features = self.feature_model.get_intermediate_layers_with_masked_feature(
                        images, self.n_last_blocks,
                    )
                
        torch.cuda.empty_cache()
        gc.collect()

        return features
    

class EncoderModel(UNet):
    def __init__(self, patch_size, n_channels, n_classes, bilinear=False):
        super(EncoderModel, self).__init__(n_channels, n_classes, bilinear)
        self.patch_size = patch_size
        self.to_grayscale_for_p2p = pth_transforms.Grayscale(num_output_channels=n_channels)
    
    def forward(self, x):
        x = self.to_grayscale_for_p2p(x)
        B, C, H, W = x.shape
        x = pypatchify.patchify_to_batches(x, (C, self.patch_size, self.patch_size), batch_dim=0)
        # print(f'x shape {x.shape}')
        x = super(EncoderModel, self).forward(x)
        x = pypatchify.unpatchify_from_batches(x, (C, H, W), batch_dim=0)
        return x

class Pipeline(nn.Module):
    def __init__(self, encoder, torch_p2p_dinov2):
        super(Pipeline, self).__init__()
        self.encoder = encoder
        self.torch_p2p_dinov2 = torch_p2p_dinov2
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.torch_p2p_dinov2(x)
        return x

