import os
import glob
import json
import numpy as np
from pathlib import Path
from typing_extensions import Self
from PIL import Image, ImageOps
# from skimage.transform import downscale_local_mean
# from skimage.color import rgb2gray

image_dir = r'/images/innoretvision/cocosearch/coco_search18_images_TP/'
train_label_file1 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
val_label_file1 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json'
# label_file2 = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split2.json'

class ImageFile:
    def __init__(self, image_path : str, img : Image = None) -> None:
        self._image_path = image_path
        self._image_dir, self._image_name = os.path.split(image_path)
        if img is None:
            # self._org_image = imread(image_path)
            self._org_image = Image.open(image_path)
            self._image = self._org_image
        else:
            self._org_image = img
            self._image = self._org_image

    def org_image(self) -> np:
        return self._org_image

    def image(self) -> np:
        return self._image

    def image_dir(self) -> str:
        return self._image_dir

    def image_name(self) -> str:
        return self._image_name

    def image_shape(self) -> tuple:
        img = np.array(self._image)
        return img.shape
    
    def resize(self, size : tuple) -> Self:
        self._image = self._image.resize(size)
        return self

    # def down_sampling(self, factors : tuple) -> Self:
        # self._image = downscale_local_mean(self._image, factors).astype('uint8')
    def down_sampling(self, size : tuple) -> Self:
        self._image = ImageOps.contain(self._image, size=size)
        return self
    
    def to_grayscale(self) -> Self:
        # self._image = rgb2gray(self._image).astype('uint8')
        self._image = ImageOps.grayscale(self._image)
        return self
    
    # def plot_image(self, saved_dir : str = '', grayscale : bool = False) -> None:
        # fig = plt.figure(figsize=(1.68, 1.05))
        # plt.axis('off')
        # if grayscale:
        #     plt.imshow(self._image, cmap=plt.cm.gray, interpolation='none')
        # else:
        #     plt.imshow(self._image, interpolation='none')

        # plt.savefig(saved_dir + '/' + self._image_name, bbox_inches='tight', pad_inches=0)
        # plt.close()
    def save_image(self, saved_dir : str = '', image_name : str = '') -> None:
        if image_name == '':
            image_name = self._image_name
        # img = Image.fromarray(self._image)
        # img.save(saved_dir + '/' + self._image_name)
        self._image.save(saved_dir + '/' + image_name)

    def crop_image(self, box : None = None) -> Image:
        if box == None:
            raise ValueError(f'Box must be a tupple of (left, upper, right, bottom)')
        img_crop = self._image.crop(box)
        return img_crop
    
    def fixation_frame(self, x : int, y : int,
                       margin_x : int, margin_y : int,
                       patch_size_x : int, patch_size_y : int,
                       image_size : tuple,) -> tuple:
        left = x + margin_x
        upper = y + margin_y
        right = left + patch_size_x
        bottom = upper + patch_size_y
        if left < 0:
            left = 0
            right = patch_size_x
        if upper < 0:
            upper = 0
            bottom = patch_size_y
        if right > image_size[0]:
            left = image_size[0] - patch_size_x
            right = image_size[0]
        if bottom > image_size[1]:
            upper = image_size[1] - patch_size_y
            bottom = image_size[1]
        # print((left, upper, right, bottom))
        return (left, upper, right, bottom)
    
    def to_fixations(self,
                        patch_size : int | tuple,
                        image_size : tuple,
                        coordinates : list = None,
                        X : np.array = None,
                        Y : np.array = None,) -> list:
        if type(patch_size) is tuple:
            patch_size_x = patch_size[0]
            patch_size_y = patch_size[1]
            margin_x = - patch_size_x // 2
            margin_y = - patch_size_y // 2
        else:
            patch_size_x = patch_size_y = patch_size
            margin_x = margin_y = - patch_size_x // 2
        fixations = []
        if coordinates is None:
            coordinates = zip(X, Y)
        for i, (x, y) in enumerate(coordinates):
            frame = self.fixation_frame(
                x, y, margin_x, margin_y, patch_size_x, patch_size_y, image_size,
            )
            img_crop = self.crop_image(frame)
            fixations.append(img_crop)
        return fixations
    
def test():
    image_path = image_dir + 'bottle/000000030947.jpg'
    image_file = ImageFile(image_path)
    # print(f'Image shape {image_file.image_shape()}')
    image_file.down_sampling((10, 10, 1)).to_grayscale().plot_image('masterthesis', grayscale=True)
    # print(f'Image shape {image_file.image_shape()}')

def downsampling_and_grayscale_patch():
    original_width, original_height = 1680, 1050
    downsampling_scale = 15
    patch_size = (16, 16)
    # images = glob.glob(os.path.join(image_dir, "*/*.jpg"))
    with open(train_label_file1) as file:
        train_data = json.load(file)
    with open(val_label_file1) as file:
        val_data = json.load(file)
    all_data = train_data + val_data
    print(f'Total image: {len(all_data)}')
    for datum in all_data:
        image_path = image_dir + datum['task'] + '/' + datum['name']
        image_file = None
        image_file = ImageFile(image_path)
        # print(f'Image shape {image_file.image_shape()}')
        new_dir = '/work/scratch/tnguyen/images/cocosearch/patches'
        print(f'saving image {image_path}')
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        image_file.down_sampling(
            (int(original_width / downsampling_scale), 
             int(original_height / downsampling_scale))
        )
        image_file.to_grayscale()
        X = np.divide(datum['X'], downsampling_scale)
        Y = np.divide(datum['Y'], downsampling_scale)
        fixations = image_file.to_fixations(
            X=X, Y=Y,
            patch_size=patch_size
        )
        saved_dir = new_dir + '/' + datum['task'] + '/' + str(datum['subject'])
        Path(saved_dir).mkdir(parents=True, exist_ok=True)
        for i, fixation in enumerate(fixations):
            image_name = str(i) + '_' + str(datum['X'][i]) + '_' + str(datum['Y'][i]) + '_' + datum['name']
            fixation.save(saved_dir + '/' + image_name)

def original_to_downsampling_and_grayscale():   
    images = glob.glob(os.path.join(image_dir, "*/*.jpg"))
    print(f'Total image: {len(images)}')
    for image_path in images:
        image_file = ImageFile(image_path)
        # print(f'Image shape {image_file.image_shape()}')
        new_dir = '/work/scratch/tnguyen' + image_file.image_dir()
        print(f'saving image to dir {new_dir}')
        Path(new_dir).mkdir(parents=True, exist_ok=True)
        # image_file.down_sampling((10, 10, 1)).plot_image(saved_dir=new_dir)
        # image_file.down_sampling((10, 10, 1)).to_grayscale().plot_image(saved_dir=new_dir, grayscale=True)
        # image_file.down_sampling((168, 105)).to_grayscale().save_image(saved_dir=new_dir)
        image_file.down_sampling((112, 70)).to_grayscale().save_image(saved_dir=new_dir)
        # print(f'Image shape {image_file.image_shape()}')

if __name__ == "__main__":
    # original_to_downsampling_and_grayscale()
    downsampling_and_grayscale_patch()
