import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from PIL import Image


label_file1 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
data_list = []
with open(label_file1) as file:
    data_list = json.load(file)
    print(len(data_list))

missed_object = {
    'bottle': 0, 'bowl': 0, 'car': 0, 'chair': 0, 'clock': 0, 'cup': 0, 
    'fork': 0, 'keyboard': 0, 'knife': 0, 'laptop': 0, 'microwave': 0, 'mouse': 0, 
    'oven': 0, 'potted plant': 0, 'sink': 0, 'stop sign': 0, 'toilet': 0, 'tv': 0
}
for datum in data_list:
    if datum['correct'] == 0:
        image_name = datum['name']
        task = datum['task']
        subject = datum['subject']
        bbox = datum['bbox']
        X = datum['X']
        Y = datum['Y']
        fix_in_bbox = 0
        for i, x in enumerate(X):
            if X[i] > bbox[0] and X[i] <= (bbox[0] + bbox[2]) and Y[i] > bbox[1] and Y[i] <= (bbox[1] + bbox[3]):
                fix_in_bbox += 1
        if fix_in_bbox == 0:
            missed_object[task] += 1
print(missed_object)


conf_matrix = np.array([
    [ 31,   1,   0,  24,   0,  34,   0,   0,   0,   0,   0,   0,   0,   9,  42,   0,   4,  25],
    [ 32,   1,   0,  22,   0,  46,   0,   1,   0,   0,   0,   0,   0,   9,  28,   0,   6,   5],
    [  6,   0,   0,  22,   0,  23,   0,   1,   0,   0,   0,   0,   0,   5,  39,   0,   3,  21],
    [  9,   0,   0,  63,   0,  26,   0,   1,   0,   0,   0,   0,   0,   4,  40,   0,  23,  92],
    [  2,   0,   0,   2,   0,  12,   0,   0,   0,   0,   0,   0,   0,   2,  67,   0,  30,  15],
    [ 19,   0,   0,  41,   0,  40,   0,   1,   1,   0,   0,   0,   0,   6,  81,   0,  22,  69],
    [  6,   1,   0,  15,   0,  26,   0,   2,   0,   0,   0,   0,   0,   4, 102,   0,  31,  43],
    [  4,   0,   0,  11,   0,  13,   1,   2,   0,   0,   0,   0,   0,   3,  51,   0,  41,  74],
    [ 13,   0,   0,  17,   0,  39,   0,   0,   0,   0,   0,   0,   0,   1,  52,   0,   6,  22],
    [  4,   0,   0,  25,   0,   9,   0,   1,   0,   0,   0,   0,   0,   8,  37,   0,   9,  37],
    [  0,   0,   0,  25,   0,  23,   1,   1,   0,   0,   0,   0,   0,   4,  42,   0,  14,  50],
    [  0,   0,   0,  10,   0,   4,   0,   8,   0,   0,   0,   0,   0,   3,  29,   0,  11,  55],
    [ 22,   0,   0,  15,   0,  14,   0,   1,   0,   0,   0,   0,   0,   9,  26,   0,   8,  15],
    [ 17,   0,   0,  43,   0,  32,   4,   1,   0,   0,   0,   0,   0,  12,  31,   0,   2,  28],
    [ 12,   0,   0,  20,   0,  42,   0,   1,   0,   0,   0,   0,   0,   7, 134,   0,  34,  40],
    [  1,   0,   0,  11,   0,   9,   0,   1,   0,   0,   0,   0,   0,   3,  34,   0,  27,  44],
    [  0,   0,   0,   2,   0,   9,   0,   0,   0,   0,   0,   0,   0,   0,  68,   0,  79,  12],
    [ 18,   0,   0,  55,   0,  37,   2,   0,   0,   0,   0,   0,   0,  10,  50,   0,  20,  97],
])
image_count = {
    'bottle': 1160, 'bowl': 980, 'car': 720, 'chair': 1766, 'clock': 830, 'cup': 1930, 
    'fork': 1610, 'keyboard': 1280, 'knife': 980, 'laptop': 860, 'microwave': 1089, 'mouse': 758, 
    'oven': 700, 'potted plant': 1070, 'sink': 1950, 'stop sign': 880, 'toilet': 1099, 'tv': 1960
}
fixation_count = {
    'bottle': 6292, 'bowl': 4960, 'car': 2959, 'chair': 7090, 'clock': 2479, 'cup': 7737, 
    'fork': 5435, 'keyboard': 3999, 'knife': 4155, 'laptop': 3222, 'microwave': 4232, 'mouse': 2420, 
    'oven': 2963, 'potted plant': 5190, 'sink': 6400, 'stop sign': 2648, 'toilet': 2845, 'tv': 6686
}
preds = np.zeros(18, dtype=int)
images = np.zeros(18, dtype=int)
fix = np.zeros(18, dtype=int)
missed = np.zeros(18, dtype=int)

fig, ax = plt.subplots()
plt.xticks(np.arange(18))
x = range(18)

for i, row in enumerate(conf_matrix):
    for j, col in enumerate(row):
        preds[j] += col
i = 0
for key, value in image_count.items():
    images[i] = value - missed_object[key]
    i += 1
i = 0
for key, value in fixation_count.items():
    fix[i] = value
    i += 1
i = 0
for key, value in missed_object.items():
    missed[i] = value * 100 / image_count[key]
    i += 1

print(preds)
ax.plot(x, preds)
ax.plot(x, images)
# ax.plot(x, fix)
# ax.plot(x, missed)
ax.set(xlabel='labels', ylabel='occurrence',
       title='Prediction distribution')
ax.grid()

# ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:grey'
# ax2.set_ylabel('percentage', color=color)  # we already handled the x-label with ax1
# ax2.plot(x, missed, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

plt.savefig('masterthesis/correct_only_label_prediction_distribution.png')
plt.close()
