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
        fixation_channel: int,
        top: float = 0.5,
    ) -> None:
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.fixation_channel = fixation_channel
        self.top = top
        self.w_featmap = self.h_featmap = img_size // patch_size
    
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
        w, h = W // self.patch_size, H // self.patch_size
        num_patches = w * h
        assert B == img_B
        if C != self.fixation_channel:
            transform = pth_transforms.Compose([
                pth_transforms.Grayscale(num_output_channels=1)
            ])
            input_images = transform(input_images)
            C = self.fixation_channel
        
        attentions = x[:, :, 0, 1:].reshape(B, NH, -1)
        # print(f"attentions shape {attentions.shape}")
        attentions = torch.sum(attentions, dim=1)
        # print(f"attentions shape {attentions.shape}")
        # return attentions

        val, idx = torch.sort(attentions, descending=True)
        # print(f'val, idx: {val}, {idx}')
        cutoff_length = int(self.top * num_patches)
        # cutoff_length = -1
        selected_idx = idx[:, :cutoff_length]
        patch_coordinates = self.find_coordinates_from_idx(selected_idx)
        # print(f"patch_coordinates shape {patch_coordinates.shape}")
        output = torch.zeros(input_images.shape)
        # for i in range(B):
        #     for c in range(C):
        #         for wj in range(W):
        #             for hj in range(H):
        #                 if torch.tensor([wj, hj]).to(self.device) in patch_coordinates[i]:
        #                     output[i][c][wj:wj+self.patch_size][hj:hj+self.patch_size] = input_images[i][c][wj:wj+self.patch_size][hj:hj+self.patch_size]
        for i in range(B):
            for coor in patch_coordinates[i]:
                x, y = int(coor[0]), int(coor[1])
                # print(f"x, y = {x, y}")
                for xi in range(x, x+self.patch_size):
                    for yi in range(y, y+self.patch_size):
                        for c in range(C):
                            output[i][c][xi][yi] = input_images[i][c][xi][yi]
        # print(f"number of non zeros: {torch.count_nonzero(output)}")
        output = output.to(self.device)
        return torch.flatten(output, start_dim=1)
    