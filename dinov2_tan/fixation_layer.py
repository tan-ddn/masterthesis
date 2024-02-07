# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import sys
import logging
import os
import torch
import math

from torch import Tensor
from torch import nn
from torchvision import transforms as pth_transforms
from PIL import Image


class Fixation(nn.Module):
    def __init__(
        self,
        device: str,
        img_size: int,
        patch_size: int,
        fixation_grayscale: bool = False,
        top: float = 0.5,
    ) -> None:
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.fixation_grayscale = fixation_grayscale
        self.top = top
        self.w_featmap = self.h_featmap = img_size // patch_size
        self.num_patches = self.w_featmap * self.h_featmap
        self.cutoff_length = int(self.top * self.num_patches)
        # self.cutoff_length = -1
        if self.fixation_grayscale:
            self.transform = pth_transforms.Compose([
                pth_transforms.Grayscale(num_output_channels=3),
                pth_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
    
    def find_coordinates_from_idx(self, id_mat):
        # y = id // self.w_featmap
        # x = id % self.w_featmap
        # return (x, y)
        x = torch.floor_divide(id_mat, self.w_featmap) * self.patch_size
        y = torch.remainder(id_mat, self.w_featmap) * self.patch_size
        x, y = x.unsqueeze(-1), y.unsqueeze(-1)
        return torch.cat((x, y), -1)
    
    def forward(self, x: Tensor, input_images: Tensor,) -> Tensor:
        B, NH, _, _ = x.shape
        img_B, C, W, H = input_images.shape
        assert (self.w_featmap == W // self.patch_size)
        assert self.h_featmap == H // self.patch_size
        assert B == img_B
        if self.fixation_grayscale:
            input_images = self.transform(input_images)
        
        attentions = x[:, :, 0, 1:].reshape(B, NH, -1)
        # print(f"attentions shape {attentions.shape}")
        attentions = torch.sum(attentions, dim=1)
        # print(f"attentions shape {attentions.shape}")
        # return attentions

        values, idx = torch.sort(attentions, descending=True)
        # print(f'val, idx: {val}, {idx}')
        # """Old and slow version"""
        # selected_idx = idx[:, :cutoff_length]
        # patch_coordinates = self.find_coordinates_from_idx(selected_idx)
        # # print(f"patch_coordinates shape {patch_coordinates.shape}")
        # output = torch.zeros(input_images.shape, device=self.device)
        # # for i in range(B):
        # #     for c in range(C):
        # #         for wj in range(W):
        # #             for hj in range(H):
        # #                 if torch.tensor([wj, hj]).to(self.device) in patch_coordinates[i]:
        # #                     output[i][c][wj:wj+self.patch_size][hj:hj+self.patch_size] = input_images[i][c][wj:wj+self.patch_size][hj:hj+self.patch_size]
        # for i in range(B):
        #     for coor in patch_coordinates[i]:
        #         x, y = int(coor[0]), int(coor[1])
        #         # print(f"x, y = {x, y}")
        #         for xi in range(x, x+self.patch_size):
        #             for yi in range(y, y+self.patch_size):
        #                 for c in range(C):
        #                     output[i][c][xi][yi] = input_images[i][c][xi][yi]
        # # print(f"number of non zeros: {torch.count_nonzero(output)}")
        # # return torch.flatten(output, start_dim=1)
        """New version"""
        cutoff_values = values[:, :self.cutoff_length]
        cutoff_values = cutoff_values[:, -1:]  # select a threshold for each image in a batch
        attn_mask = torch.where(attentions > cutoff_values, 1., 0.)
        attn_mask = attn_mask.unsqueeze(1).reshape(B, 1, self.w_featmap, self.h_featmap)
        # print(f"attn_mask shape {attn_mask.shape}")
        attn_mask = torch.nn.functional.interpolate(attn_mask, size=(W, H), mode='nearest-exact')
        # print(f"attn_mask shape after interpolate {attn_mask.shape}")
        output = input_images * attn_mask.expand(-1, 3, -1, -1)
        del attentions, values, idx, attn_mask
        return output
    

def get_mask_from_top_attn(attn: Tensor, cutoff_length: int) -> Tensor:
    B, NH, _, _ = attn.shape
        
    attentions = attn[:, :, 0, 1:].reshape(B, NH, -1)
    # print(f"attentions shape {attentions.shape}")
    attentions = torch.sum(attentions, dim=1)
    # attn_length = int(math.sqrt(attentions.shape[-1]))
    # print(f"attentions shape {attentions.shape}")
    # return attentions

    values, idx = torch.sort(attentions, descending=True)
    """New version"""
    cutoff_values = values[:, :cutoff_length]
    cutoff_values = cutoff_values[:, -1:]  # select a threshold for each image in a batch
    attn_mask = torch.where(attentions > cutoff_values, 1., 0.)
    # attn_mask = attn_mask.unsqueeze(1).reshape(B, 1, attn_length, attn_length)
    # print(f"attn_mask shape {attn_mask.shape}")
    del attentions, values, idx, cutoff_values
    return attn_mask
    

if __name__ == '__main__':
    """For testing purpose"""
    
    attentions = torch.tensor([
        [15., 0., 13., 8.],
        [7., 2., 0., 10.],
    ])
    cutoff_values = torch.tensor([
        [5.],
        [0.],
    ])
    attn_mask = torch.where(attentions > cutoff_values, 1., 0.)
    print(f"attn_mask {attn_mask}")

    # attn_mask = torch.tensor([[[
    #     [0., 0., 1., 1.],
    #     [0., 0., 0., 1.],
    #     [0., 1., 1., 1.],
    #     [1., 0., 0., 0.],
    # ]]])
    # print(f"attn_mask {attn_mask}")
    # print(f"attn_mask shape {attn_mask.shape}")
    # attn_mask = torch.nn.functional.interpolate(attn_mask, scale_factor=(3, 3), mode='nearest-exact')
    # print(f"attn_mask {attn_mask}")
    # print(f"attn_mask shape {attn_mask.shape}")
