# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision.transforms import v2


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(v2.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    grayscale: bool = False,
) -> v2.Normalize:
    if grayscale:
        return v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    return v2.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    grayscale: bool = False,
    norm = 'norm',
    pulse2percept = False,
):
    if pulse2percept:
        transforms_list = []
        if hflip_prob > 0.0:
            transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
        if norm == 'norm':
            transforms_list.extend(
                [
                    MaybeToTensor(),
                    make_normalize_transform(mean=mean, std=std, grayscale=grayscale),
                ]
            )
        else:
            transforms_list.extend(
                [
                    MaybeToTensor(),
                ]
            )
        return v2.Compose(transforms_list)
    transforms_list = [v2.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    if grayscale:
        transforms_list.append(v2.Grayscale(num_output_channels=3))
    if norm == 'norm':
        transforms_list.extend(
            [
                MaybeToTensor(),
                make_normalize_transform(mean=mean, std=std, grayscale=grayscale),
            ]
        )
    else:
        transforms_list.extend(
            [
                MaybeToTensor(),
            ]
        )
    return v2.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=v2.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
    grayscale: bool = False,
    norm = 'norm',
    pulse2percept = False,
) -> v2.Compose:
    if pulse2percept:
        if norm == 'norm':
            transforms_list = [
                MaybeToTensor(),
                make_normalize_transform(mean=mean, std=std, grayscale=grayscale),
            ]
        else:
            transforms_list = [
                MaybeToTensor(),
            ]
        return v2.Compose(transforms_list)
    if norm == 'norm':
        transforms_list = [
            v2.Resize(resize_size, interpolation=interpolation),
            v2.CenterCrop(crop_size),
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std, grayscale=grayscale),
        ]
        if grayscale:
            transforms_list = [
                v2.Resize(resize_size, interpolation=interpolation),
                v2.CenterCrop(crop_size),
                v2.Grayscale(num_output_channels=3),
                MaybeToTensor(),
                make_normalize_transform(mean=mean, std=std, grayscale=grayscale),
            ]
    else:
        transforms_list = [
            v2.Resize(resize_size, interpolation=interpolation),
            v2.CenterCrop(crop_size),
            MaybeToTensor(),
        ]
        if grayscale:
            transforms_list = [
                v2.Resize(resize_size, interpolation=interpolation),
                v2.CenterCrop(crop_size),
                v2.Grayscale(num_output_channels=3),
                MaybeToTensor(),
            ]
    return v2.Compose(transforms_list)
