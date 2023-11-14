import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image

image_dir = r'/work/scratch/tnguyen/images/innoretvision/cocosearch/coco_search18_images_TP/'
label_file1 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
# label_file2 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split2.json'

data_list = []
with open(label_file1) as file:
    data_list = json.load(file)
    print(len(data_list))

# with open(label_file2) as file:
#     data_list += json.load(file)
#     print(len(data_list))

incorrect_data = []
for datum in data_list:
    if datum['correct'] == 0:
        image_name = datum['name']
        task = datum['task']
        subject = datum['subject']
        bbox = datum['bbox']
        X = datum['X']
        Y = datum['Y']
        image_file = image_dir + task + '/' + image_name
        incorrect_data.append({
            'image_name': image_name,
            'task': task,
            'subject': subject,
            'bbox': bbox,
            'X': X,
            'Y': Y,
            'image_file': image_file
        })

print(len(incorrect_data))
# for item in incorrect_data:
#     # item = incorrect_data[700]
#     imgplot = plt.imread(item['image_file'])
#     fig, ax = plt.subplots(figsize=(16.8/4, 10.5/4))
#     ax.imshow(imgplot)
#     ax.plot(item['X'], item['Y'], 'bo')

#     rect = patches.Rectangle((item['bbox'][0], item['bbox'][1]), item['bbox'][2], item['bbox'][3], edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

#     plt.savefig('plots/task_' + item['task'] + '_' + str(item['subject']) + '_' + item['image_name'])
#     plt.close()

"""Draw an example image"""
item = incorrect_data[10]
factor = 10
factor = 15
item['bbox'][0] /= factor
item['bbox'][1] /= factor
item['bbox'][2] /= factor
item['bbox'][3] /= factor

imgplot = plt.imread(item['image_file'])
fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(imgplot, cmap=plt.cm.gray)

for i, x in enumerate(item['X']):
    x = x / factor
    y = item['Y'][i] / factor
    fix_rect = None
    fix_rect = patches.Rectangle((x-8, y-8), 16, 16, edgecolor='b', facecolor='none')
    ax.add_patch(fix_rect)

bbox_rect = patches.Rectangle((item['bbox'][0], item['bbox'][1]), item['bbox'][2], item['bbox'][3], edgecolor='r', facecolor='none')
ax.add_patch(bbox_rect)

plt.savefig('plots/g_15_task_' + item['task'] + '_' + str(item['subject']) + '_' + item['image_name'], bbox_inches='tight', pad_inches=0)
plt.close()
