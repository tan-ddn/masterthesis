import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.io as tv_io
from pathlib import Path

TRAIN_LABEL_FILE = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
VAL_LABEL_FILE = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json'
IMAGE_DIR = r'/work/scratch/tnguyen/images/innoretvision/cocosearch/coco_search18_images_TP/'

class FixationDataset(torch.utils.data.Dataset):
    def __init__(self, data : None,
                 image_dir : str,
                 max_fix_length : int,
                 channels : int,
                 patch_size : tuple,
                 device : str, ):
        super().__init__()
        self.data = data
        self.image_dir = image_dir
        self.max_fix_length = max_fix_length
        self.channels = channels
        self.patch_size = patch_size
        self.device = device

        self.labels = {
            "bottle": 0, "bowl": 1, "car": 2, "chair": 3, "clock": 4, "cup": 5,
            "fork": 6, "keyboard": 7, "knife": 8, "laptop": 9, "microwave": 10, "mouse": 11,
            "oven": 12, "potted plant": 13, "sink": 14, "stop sign": 15, "toilet": 16, "tv": 17,
        }

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        """Load data and get label"""
        datum = self.data[index]
        patch_dir = self.image_dir + datum['task'] + '/' + str(datum['subject']) + '/'
        fixations = torch.zeros(
            (self.max_fix_length, self.channels, self.patch_size[0], self.patch_size[1]),
            device=self.device,
        )
        # fixations = []
        for i, (x, y) in enumerate(zip(datum['X'], datum['Y'])):
            patch_name = str(i) + '_' + str(x) + '_' + str(y) + '_' + datum['name']
            patch = tv_io.read_image(
                patch_dir+patch_name,
                mode=tv_io.ImageReadMode.GRAY
            )  # patch shape = (1, 16, 16)
            fixations[i] = patch
            # fixations.append(patch)
        # fixations = torch.stack(fixations, dim=0).to(self.device)

        if datum['task'] in self.labels:
            label = self.labels[datum['task']]
        else:
            label = -1
        return fixations, label
    
    # def __getitem__(self, index):
    #     """Generates one sample of data"""
    #     """Load data and get label"""
    #     datum = self.data[index]
    #     image_path = self.image_dir + datum['task'] + '/' + datum['name']
    #     img = tv_io.read_image(image_path, mode=tv_io.ImageReadMode.GRAY)  # img shape = (channels, height, width)
    #     # fixations = self.image2fixations(img, datum['X'], datum['Y'])  # fixations shape = (no_fixations, channels, height, width)
    #     fixation_locations = torch.zeros((self.max_fix_length, 2), device=self.device)
    #     fixation_locations = torch.sub(fixation_locations, 1)
    #     for i, (x, y) in enumerate(zip(datum['X'], datum['Y'])):
    #         fixation_locations[i] = torch.tensor([x, y], device=self.device)
    #     # print(f'Fixation locations {fixation_locations}')
    #     # print(f'Fixation locations shape {fixation_locations.shape}')

    #     if datum['task'] in self.labels:
    #         label = self.labels[datum['task']]
    #     else:
    #         label = -1
    #     # return fixations, label
    #     return img, fixation_locations, label
    