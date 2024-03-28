import os
import glob
import torch
import argparse
import cv2
import numpy as np
from torchvision import transforms as tv_transforms
from pulse2percept_layer import build_p2p_model_and_implant, batch_imgs_generator, image2percept

RANGE = (0, 1001)

def get_left_images(source_dir_path, target_dir_path, range):
    total_target_files = 0
    total_left_files = {}
    counter = 0
    for root, _, source_files in os.walk(source_dir_path):
        if counter > range[1]:
            break
        if root == source_dir_path or counter < range[0]:
            counter += 1
            continue
        # print(f"root {root}")
        folder_name = root.split("/")[-1]
        target_folder = os.path.join(target_dir_path, folder_name)
        target_files = next(os.walk(target_folder))[2]
        left_files = [os.path.join(root, f) for f in source_files if f not in target_files]
        total_target_files += len(target_files)
        counter += 1
        if len(left_files):
            print(f'Folder {folder_name}: Imnet files {len(source_files)} - P2P files {len(target_files)} = {len(left_files)}')
            total_left_files[folder_name] = left_files
    print(f'Total files in p2p {total_target_files}')
    return total_left_files

def main(files):
    parser = argparse.ArgumentParser('Create percept from pulse2percept')
    parser.add_argument("--image_path", default='/images/PublicDatasets/imagenet/', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=224, type=int, help="Resize image.")
    parser.add_argument('--patch_size', default=14, type=int, help='Patch resolution of the model.')
    parser.add_argument('--batch_size', default=4096, type=int, help='Batch size.')
    parser.add_argument("--range", default=(0, 1), type=int, nargs="+", help="Range of images in dataset.")
    parser.add_argument('--output_dir', default='/images/innoretvision/eye/imagenet_patch/p2p/', help='Path where to save visualizations.')
    parser.add_argument("--time_track", default=False, type=bool, help="Track time during the pulse2percept.")
    args = parser.parse_args()

    args.output_dir = args.output_dir + r"train" + "/"
    
    print(f'range {args.range}')
    start = args.range[0]
    end = args.range[1]
    files = files[int(start):int(end)]
    print(f'total files {len(files)}')
        
    """Prepare p2p_model and implant"""
    p2p_patch_size = args.patch_size
    p2p_model, square16_implant = build_p2p_model_and_implant(size=p2p_patch_size)

    for saved_filenames, imgs in batch_imgs_generator(files, args):
        print(f'{len(imgs)}')
        percept = image2percept(imgs, args, p2p_patch_size, p2p_model, square16_implant)
        for i in range(len(percept)):
            if os.path.isfile(saved_filenames[i]):
                continue
            # save_image(percept[i], saved_filenames[i])
            cv2.imwrite(saved_filenames[i], percept[i]) 
            print(f'{saved_filenames[i]}')

def read_image_max_min(file_path):
    img = cv2.imread(file_path)
    img = np.array(img)
    transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = transform(img)
    print(f'max min {torch.max(img), torch.min(img)}')


if __name__ == '__main__':
    imnet_dir_path = r'/images/PublicDatasets/imagenet/train/'
    p2p_dir_path = r'/images/innoretvision/eye/imagenet_patch/p2p/train/'

    # imnet_dirs = []
    # for dirnames in os.listdir(imnet_dir_path):
    #     if os.path.isdir(os.path.join(imnet_dir_path, dirnames)):
    #         imnet_dirs.append(dirnames)
    # imnet_dirs = tuple(imnet_dirs)
    # print(f'Number of dirs in imnet {len(imnet_dirs)}')

    # p2p_dirs = []
    # for dirnames in os.listdir(p2p_dir_path):
    #     if os.path.isdir(os.path.join(p2p_dir_path, dirnames)):
    #         p2p_dirs.append(dirnames)
    # p2p_dirs = tuple(p2p_dirs)
    # print(f'Number of dirs in p2p {len(p2p_dirs)}')

    # p2p_total = 0
    # for imnet_dir in imnet_dirs:
    #     imnet_count = 0
    #     p2p_count = 0
    #     imnet_sub_dir = imnet_dir_path + str(imnet_dir) + '/'
    #     p2p_sub_dir = p2p_dir_path + str(imnet_dir) + '/'
    #     # print(f'Sub dir: {imnet_sub_dir} and {p2p_sub_dir}')
    #     imnet_count += len(next(os.walk(imnet_sub_dir))[2])
    #     p2p_count += len(next(os.walk(p2p_sub_dir))[2])
    #     p2p_total += p2p_count
    #     if p2p_count != imnet_count:
    #         print(f'Folder {imnet_dir}: Imnet files {imnet_count} - P2P files {p2p_count}')
    # print(f'Total files in p2p {p2p_total}')

    # """Check images that haven't been processed by p2p to create p2p dataset"""
    # left_images = get_left_images(imnet_dir_path, p2p_dir_path, RANGE)
    # left_images = list(left_images.values())
    # left_images = [x for sublist in left_images for x in sublist]
    # left_images = list(reversed(left_images))

    # print(left_images[0])
    # main(left_images)

    """Check max and min values of the images in p2p dataset"""
    files = glob.glob(os.path.join(p2p_dir_path, "*/*.jpg"))
    read_image_max_min(files[12345])
