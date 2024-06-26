import argparse
import logging
import os
import glob
import cv2
import gc
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
from torchvision import transforms as pth_transforms


sys.path.append("/home/students/tnguyen/masterthesis")

from p2p.torch_p2p_dinov2_masked_model import image2percept, build_p2pmodel_and_implant, AxonMapSpatialModifiedModule
from unet.unet_model import UNet, UNet2


def down_sampled_image2percept(img, desired_size, decoder, chunk_length=64):
    # crop_start = (decoder.percept_shape[0] - p2p_patch_size) // 2
    # crop_end = crop_start + p2p_patch_size
    margin  = int(decoder.percept_shape[0] * 0.1)
    crop_start = int(decoder.percept_shape[0] * 0.25) - margin
    crop_end = crop_start + int(decoder.percept_shape[0] * 0.5) + (2 * margin)
    crop_size = crop_end - crop_start

    # print(f"img info {img.shape}, {img.max()}, {img.min()}")
    img = torch.flatten(img, start_dim=1)

    # percept = torch.zeros((patches.shape[0], p2p_patch_size, p2p_patch_size), device=img.device)
    percept = torch.zeros((img.shape[0], crop_size, crop_size), device=img.device)
    # for i, patch in enumerate(patches):
    #     patch_percept_rescaled = decoder([patch.unsqueeze(0), phis.unsqueeze(0)])[:, crop_start:crop_end, crop_start:crop_end]
    #     # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

    #     # take notes of the max of the patch
    #     patch_max = torch.max(patch)
    #     if patch_max > 0.0:
    #         patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

    #     percept[i] = patch_percept_rescaled

    # percept = decoder([patches, torch.tile(phis, (len(patches), 1))])[:, crop_start:crop_end, crop_start:crop_end]
    # # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))
    
    if chunk_length > 0:
        chunk_length = chunk_length
        list_of_patches = torch.split(img, chunk_length)
        # print(f"chunk shape {list_of_patches[0].shape}")
        for i, patch_chunk in enumerate(list_of_patches):
            # print(f'phis shape {phis.shape}')
            patch_percept_rescaled = decoder(patch_chunk)[:, crop_start:crop_end, crop_start:crop_end]
            # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

            # # take notes of the max of the patch
            # patch_max = torch.max(patch_chunk)
            # if patch_max > 0.0:
            #     patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

            percept[i*chunk_length:((i+1)*chunk_length)] = patch_percept_rescaled
            del patch_percept_rescaled
    else:
        percept = decoder(img)[:, crop_start:crop_end, crop_start:crop_end]

    # take notes of the max of the patch
    patch_max = torch.max(img)
    if patch_max > 0.0:
        percept = (percept - percept.min())/(percept.max() - percept.min())*patch_max

    percept = percept.unsqueeze(1)
    # print(f"percept shape {percept.shape}")
    
    """Resize percept back to desired shape"""
    percept = F.interpolate(percept, size=(desired_size, desired_size), mode='nearest-exact')
    # print(f"percept shape {percept.shape}")

    del img

    return torch.tile(percept, (1, 3, 1, 1))


class ModelWithMaskedLastFeature(nn.Module):
    def __init__(self, feature_model, top_attention, n_last_blocks, autocast_ctx, args):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.top_attention = top_attention
        if not args.no_eval:
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
            if not args.no_eval:
                self.decoder.eval()
            self.percept_norm_for_dino = pth_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def get_features(self, images):
        with self.autocast_ctx():
            # print(f'top_attention {self.feature_model.top_attention}')
            if self.args.torch_p2p:
                if self.args.down_sampling:
                    percepts = down_sampled_image2percept(images, self.args.image_size, self.decoder, self.args.chunk_length)
                else:
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

    def forward(self, images):
        if self.args.grad_mode == 'none':
            return self.get_features(images=images)
        elif self.args.grad_mode == 'inference':
            with torch.inference_mode():
                return self.get_features(images=images)
        elif self.args.grad_mode == 'no_grad':
            with torch.no_grad():
                return self.get_features(images=images)
    

class EncoderModel(nn.Module):
    def __init__(self, patch_size, n_channels, n_classes, bilinear=False, encoder='unet', down_sampling=False):
        # super(EncoderModel, self).__init__(n_channels, n_classes, bilinear)
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(patch_size**2, patch_size**2)
        )
        self.unet = UNet(n_channels, n_classes, bilinear)
        self.unet2 = UNet2(n_channels, n_classes, bilinear)
        self.encoder = encoder
        self.down_sampling = down_sampling
        self.patch_size = patch_size
        self.to_grayscale_for_p2p = pth_transforms.Grayscale(num_output_channels=n_channels)
    
    def forward(self, x):
        x = self.to_grayscale_for_p2p(x)
        B, C, H, W = x.shape

        if self.encoder == 'none':  # if no encoder, no need to do anything
            return x
        
        if not self.down_sampling:  # patchify the image into smaller patches if the image is not down sampled yet
            x = pypatchify.patchify_to_batches(x, (C, self.patch_size, self.patch_size), batch_dim=0)
        # print(f'x shape {x.shape}')
        # x = super(EncoderModel, self).forward(x)
        if self.encoder == 'unet':
            x = self.unet.forward(x)
        elif self.encoder == 'unet2':
            x = self.unet2.forward(x)
        else:
            x = self.linear(x)
            x = x.view(-1, C, self.patch_size, self.patch_size)
        if not self.down_sampling:
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

if __name__ == '__main__':
    """Test view()"""
    x = torch.tensor([
        [[0, 1, 2, 3]],
        [[4, 5, 6, 7]],
        [[8, 9, 10, 11]]
        ])
    print(x.shape)
    x = x.view(-1, 1, 2, 2)
    print(x)

    """Test down_sampled_image2percept()"""
