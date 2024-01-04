import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.io as tv_io
from pathlib import Path

COCO_TP_TRAIN_LABEL_FILE = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
COCO_TP_VAL_LABEL_FILE = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json'
COCO_TP_IMAGE_DIR = r'/work/scratch/tnguyen/images/innoretvision/cocosearch/coco_search18_images_TP/'

def read_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

def coco18tp_data(args, defaultValues):
    train_data = read_data(defaultValues['train_label_file'])
    train_data = train_data[0: args.data_limit]
    train_dataset = FixationDataset(
        data=train_data,
        image_dir=defaultValues['image_dir'],
        max_fix_length=defaultValues['max_fix_length'],
        channels=args.num_channel,
        patch_size=defaultValues['patch_size'],
        device=args.device,
    )  
    val_data = read_data(defaultValues['val_label_file'])
    val_data = val_data[0: args.data_limit]
    val_dataset = FixationDataset(
        data=val_data,
        image_dir=defaultValues['image_dir'],
        max_fix_length=defaultValues['max_fix_length'],
        channels=args.num_channel,
        patch_size=defaultValues['patch_size'],
        device=args.device,
    )  
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=False,
        # num_workers=8,
    )
    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=True,
        pin_memory=False,
        # num_workers=8,
    )
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
    }
    return trainloader, valloader, dataset_sizes


class FixationDataset(torch.utils.data.Dataset):
    def __init__(self, data : None,
                 image_dir : str,
                 max_fix_length : int,
                 channels : int,
                 patch_size : tuple,
                 device : str = 'cpu', ):
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
    
if __name__ == "__main__":
    image_dir = r'/work/scratch/tnguyen/images/cocosearch/patches/'
    train_data = read_data(COCO_TP_TRAIN_LABEL_FILE)
    
    '''Count fixations for each task'''
    fixation_counts = {
        "bottle": 0, "bowl": 0, "car": 0, "chair": 0, "clock": 0, "cup": 0,
        "fork": 0, "keyboard": 0, "knife": 0, "laptop": 0, "microwave": 0, "mouse": 0,
        "oven": 0, "potted plant": 0, "sink": 0, "stop sign": 0, "toilet": 0, "tv": 0,
    }
    image_counts = {
        "bottle": 0, "bowl": 0, "car": 0, "chair": 0, "clock": 0, "cup": 0,
        "fork": 0, "keyboard": 0, "knife": 0, "laptop": 0, "microwave": 0, "mouse": 0,
        "oven": 0, "potted plant": 0, "sink": 0, "stop sign": 0, "toilet": 0, "tv": 0,
    }
    for datum in train_data:
        label = datum['task']
        num_fixations = len(datum['X'])
        fixation_counts[label] += int(num_fixations)
        image_counts[label] += 1
    print(f'fixation count: {fixation_counts}')
    print(f'image count: {image_counts}')
