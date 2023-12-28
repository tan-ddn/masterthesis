# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
import gc
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from dino import utils
from dino import vision_transformer as vits
from utils.image_processing import ImageFile


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def img_generator(files):
    for image_file in files:
        print(f'{image_file}')
        with open(image_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        transform = pth_transforms.Compose([
            pth_transforms.Resize(args.image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        yield image_file, transform(img)


def find_coordinates_from_idx(columns, id):
    y = id // columns
    x = id % columns
    return (x, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/home/students/tnguyen/masterthesis/dino/dino_deitsmall8_pretrain.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default='/images/PublicDatasets/imagenet/', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument("--range", default=(0, 1), type=int, nargs="+", help="Range of images in dataset.")
    parser.add_argument('--output_dir', default='/work/scratch/tnguyen/images/imagenet/patches/', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    # if args.image_path is None:
    #     # user has not specified any image - we use our own image
    #     print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
    #     print("Since no image path have been provided, we take the first image in our paper.")
    #     response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
    #     img = Image.open(BytesIO(response.content))
    #     img = img.convert('RGB')
    # elif os.path.isfile(args.image_path):
    #     with open(args.image_path, 'rb') as f:
    #         img = Image.open(f)
    #         img = img.convert('RGB')
    # else:
    #     print(f"Provided image path {args.image_path} is non valid.")
    #     sys.exit(1)

    files = glob.glob(os.path.join(args.image_path, "train/*/*.jpg"))  # 1281167
    if args.image_path == '/images/PublicDatasets/imagenet_shared/':
        files = glob.glob(os.path.join(args.image_path, "val/*/*"))  # 50000
    # files = glob.glob(os.path.join(args.image_path, "train/*/n02113023_7135.jpg"))
    # random.shuffle(files)
    print(f'range {args.range}')
    start = args.range[0]
    end = args.range[1]
    files = files[int(start):int(end)]
    print(f'total files {len(files)}')
    # for image_file in files:
    #     print(f'{image_file}')
    #     with open(image_file, 'rb') as f:
    #         img = Image.open(f)
    #         img = img.convert('RGB')
    #     transform = pth_transforms.Compose([
    #         pth_transforms.Resize(args.image_size),
    #         pth_transforms.ToTensor(),
    #         pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     ])
    #     img = transform(img)
    # Generator
    for image_file, img in img_generator(files):
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size
        # print(f'{w_featmap}, {h_featmap}')

        attentions = model.get_last_selfattention(img.to(device))

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        attentions = torch.sum(attentions, dim=0)
        # print(f"Attention shape {attentions.shape}")

        # if args.threshold is not None:
        #     # we keep only a certain percentage of the mass
        #     val, idx = torch.sort(attentions)
        #     val /= torch.sum(val, dim=1, keepdim=True)
        #     cumval = torch.cumsum(val, dim=1)
        #     th_attn = cumval > (1 - args.threshold)
        #     idx2 = torch.argsort(idx)
        #     for head in range(nh):
        #         th_attn[head] = th_attn[head][idx2[head]]
        #     th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        #     # interpolate
        #     th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(attentions, dim=0, keepdim=True)
            print(f'val, idx: {val}, {idx}')
            cumval = torch.cumsum(val, dim=0)
            print(f'cumval {cumval}')
            th_attn = cumval > (1 - args.threshold)
            print(f'th_attn {th_attn}')
            idx2 = torch.argsort(idx)
            print(f'idx2 {idx2}')
            th_attn = th_attn[idx2]
            th_attn = th_attn.reshape(1, w_featmap, h_featmap).float()

            """Get the coordinates of attention"""
            patch_centers = []
            th_attn_map = th_attn[0].cpu().numpy()
            print(f'attn map shape {th_attn_map.shape}')
            print(f'attn map {th_attn_map}')
            coordinates = list(zip(*np.where(th_attn_map >= 0.5)))
            patch_centers.append(coordinates)
            # print(f'patch_centers length {len(patch_centers)}')
            patch_centers = [(x, y) for sublist in patch_centers for (y, x) in sublist]  # flatten the list
            patch_centers = list(set(patch_centers))  # remove tuple duplicates
            random.shuffle(patch_centers)
            patch_centers = patch_centers[0:10]  # choose only first 10 centers
            for i, (x, y) in enumerate(patch_centers):
                resized_center = (x * args.patch_size, y * args.patch_size)
                patch_centers[i] = resized_center
            print(f'patch coordinates {patch_centers}')

            # interpolate
            th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()
        else:
            val, idx = torch.sort(attentions, descending=True)
            # print(f'val, idx: {val}, {idx}')
            """Get the coordinates of attention"""
            coordinates = []
            for id in idx.cpu().numpy():
                coord = find_coordinates_from_idx(w_featmap, id)
                coordinates.append(coord)
            reserved_coordinates = []
            patch_centers = []
            for (x, y) in coordinates:
                if len(patch_centers) > 9:
                    break
                search_coord = None
                search_coord = next((i for i, c in enumerate(reserved_coordinates) if c[0] == x and c[1] == y), None)
                if search_coord is None:
                    patch_centers.append((x, y))
                    # block the coordinate and the surrounding coordinates as well
                    for m in range(-5, 5):
                        for n in range(-5, 5):
                            neighbor_coords = (x + m, y + n)
                            reserved_coordinates.append(neighbor_coords)              

            # print(f'patch_centers length {len(patch_centers)}')
            for i, (x, y) in enumerate(patch_centers):
                resized_center = (x * args.patch_size, y * args.patch_size)
                patch_centers[i] = resized_center
            # print(f'patch coordinates {patch_centers}')

        # attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = attentions.reshape(1, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

        image_dir, image_name = os.path.split(image_file)  # image_dir ~ /images/PublicDatasets/imagenet/train/n02823428
        image_dir = image_dir.split('/')
        image_dir = image_dir[-2:]
        image_name = image_name.split('.')
        saved_dir = args.output_dir + image_dir[0] + '/' + image_dir[1] + '/'
        os.makedirs(saved_dir, exist_ok=True)

        # save attentions heatmaps
        # os.makedirs(args.output_dir, exist_ok=True)
        # torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
        # for j in range(nh):
        #     fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        #     plt.imsave(fname=fname, arr=attentions[j], format='png')
        #     print(f"{fname} saved.")

        # torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(saved_dir, image_name[0] + '.' + image_name[1]))
        # fname = os.path.join(saved_dir, image_name[0] + "attn-head" + ".png")
        # plt.imsave(fname=fname, arr=attentions[0], format='png')

        if args.threshold is not None:
            image = skimage.io.imread(os.path.join(saved_dir, image_name[0] + '.' + image_name[1]))
            # for j in range(nh):
            #     display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)
            # print(f'th_attn shape {th_attn.shape}')
            # print(f'th_attn {th_attn}')
            display_instances(image, th_attn[0], fname=os.path.join(saved_dir, image_name[0] + "mask_th" + str(args.threshold) + "_head" +".png"), blur=False)

        """Downsampling image to 48x48 and perform grayscale"""
        image_obj = ImageFile(image_file)
        image_obj.resize(args.image_size)
        image_obj.to_grayscale()
        # image_obj.save_image(saved_dir)
        fixations = image_obj.to_fixations(
            patch_size=(48, 48),
            image_size=args.image_size,
            coordinates=patch_centers,
        )
        transform = pth_transforms.Compose([
            pth_transforms.Resize((16, 16)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.5), (0.5)),
        ])
        fixations_tensor = torch.zeros((len(fixations), 1, 16, 16))
        for i, fixation in enumerate(fixations):
            # fixation_name = image_name[0] + '_' + str(i) + '.' + image_name[1]
            # fixation_path = saved_dir + fixation_name
            # # print(f'{fixation_path}')
            # # fixation.save(fixation_path)
            # fixation_resized = fixation.resize((16, 16))
            # fixation_resized_name = image_name[0] + '_' + str(i) + '_16x16.' + image_name[1]
            # fixation_resized.save(saved_dir + fixation_resized_name)
            fixation = transform(fixation)
            fixations_tensor[i] = fixation
            torch.save(fixations_tensor, saved_dir + image_name[0] + f'_16x16.pt')
        # saved_fixations = torch.load(saved_dir + image_name[0] + f'_16x16.pt')
        # print(f'saved_fixations shape {saved_fixations.shape}')

        del attentions
        del img, image_obj, fixation
        torch.cuda.empty_cache()
        gc.collect()
