import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image

image_dir = r'/images/innoretvision/cocosearch/coco_search18_images_TP/'
label_file1 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
label_file1 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json'
# label_file2 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split2.json'

data_list = []
with open(label_file1) as file:
    data_list = json.load(file)
    print(len(data_list))

# with open(label_file2) as file:
#     data_list += json.load(file)
#     print(len(data_list))

max_fix_length = 0
incorrect_data = []
incorrect_data_no_fixation_in_bbox = []
for datum in data_list:
    if int(datum['length']) > max_fix_length:
        max_fix_length = int(datum['length']) 
    if datum['correct'] == 0:
        image_name = datum['name']
        task = datum['task']
        subject = datum['subject']
        bbox = datum['bbox']
        X = datum['X']
        Y = datum['Y']
        image_file = image_dir + task + '/' + image_name
        item = {
            'image_name': image_name,
            'task': task,
            'subject': subject,
            'bbox': bbox,
            'X': X,
            'Y': Y,
            'image_file': image_file
        }
        incorrect_data.append(item)
        fix_in_bbox = 0
        for i, x in enumerate(X):
            if X[i] > bbox[0] and X[i] <= (bbox[0] + bbox[2]) and Y[i] > bbox[1] and Y[i] <= (bbox[1] + bbox[3]):
                fix_in_bbox += 1
        if fix_in_bbox == 0:
            incorrect_data_no_fixation_in_bbox.append(item)

print(f'Max fix length {max_fix_length}')
print(len(incorrect_data))
print(len(incorrect_data_no_fixation_in_bbox))
