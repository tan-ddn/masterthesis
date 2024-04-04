import os
import glob
import numpy as np
import torch
import argparse
import pypatchify
import gc
import threading
import math
import time
import cv2
import matplotlib.pyplot as plt
# import torchvision.io.image as tv_io_image
from PIL import Image
from pathlib import Path
from numpy import pi
# from torchvision.transforms import v2
from torchvision import transforms as tv_transforms
from torchvision.io import read_image
from torchvision.utils import save_image
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem
from pulse2percept.stimuli import ImageStimulus

SPACE = 300

def build_p2p_model_and_implant(size, axlambda=100, rho=150, range_limit=13):
    p2p_model = AxonMapModel(
        axlambda=axlambda, rho=rho, 
        xrange=(-range_limit, range_limit), yrange=(-range_limit, range_limit), xystep=1
    )
    p2p_model.build()

    square16_implant = ProsthesisSystem(earray=ElectrodeGrid((size, size), SPACE))
    
    return p2p_model, square16_implant

def get_percept(image, p2p_model, implant):
    image_stim = ImageStimulus(image).rgb2gray()
    implant.stim = image_stim
    percept = p2p_model.predict_percept(implant)
    return percept

def image2percept(image, args, p2p_patch_size, p2p_model, square16_implant):
    crop_start = (p2p_patch_size // 2) - 1
    crop_end = crop_start + p2p_patch_size

    """Full image to patches (new method)"""
    # num_patches = (args.image_size // args.patch_size) ** 2
    # print(f'image shape {image.shape}')  # shape (N, C, args.image_size, args.image_size)
    patches = pypatchify.patchify_to_batches(image, (p2p_patch_size, p2p_patch_size, 3), batch_dim=0)
    # patches = patches.permute(0, 2, 3, 1).cpu().detach().numpy()
    # print(f'patches shape {patches.shape}')  # shape (N * num_patches, args.patch_size, args.patch_size, C)
    # percept = []
    # for patch in patches:
    #     patch_percept = get_percept(patch, size=(args.patch_size, args.patch_size))
    #     # print(f'patch_percept shape {patch_percept.data.shape}')
    #     # percept.append(patch_percept.data)
    #     crop = patch_percept.data[30:91, 30:91, :]
    #     percept.append(crop)

    def percept_task(no_crop, start_index, patches, percept, thread_index):
        # print(f"Thread {thread_index} starts")
        for i, patch in enumerate(patches):
            percept_index = start_index + i
            patch_percept = get_percept(patch, p2p_model, square16_implant)
            # print(f'patch_percept shape {patch_percept.data.shape}')  # (27, 27, 1)
            if no_crop:
                # print(f'patch_percept shape {patch_percept.data.shape}')
                percept[percept_index] = patch_percept.data
            else:
                crop_percept = patch_percept.data[crop_start:crop_end, crop_start:crop_end, :]
                percept[percept_index] = crop_percept
            # if (thread_index == 1) and ((percept_index % 10) == 0):
            #     print(f"percept_index {percept_index}")

    """Threading"""
    num_thread = min(32768, int(math.sqrt(patches.shape[0])))
    print(f'num_thread {num_thread}')
    # percept = np.zeros((patches.shape[0], 56, 56, 1))
    percept = [None] * patches.shape[0]
    list_of_patches = np.array_split(patches, num_thread)
    threads = [None] * num_thread
    next_chunk_start_index = 0
    for i in range(num_thread):
        threads[i] = threading.Thread(target=percept_task, args=(args.no_crop, next_chunk_start_index, list_of_patches[i], percept, i, ))
        threads[i].start()
        next_chunk_start_index += len(list_of_patches[i])
    for i in range(num_thread):
        threads[i].join()

    percept = np.array(percept)
    # percept = torch.tensor(np.array(percept), device=args.device).permute(0, 3, 1, 2)  # move the channel to the second dim
    # # print(f'percept shape {percept.shape}')
    # percept = torch.nn.functional.interpolate(percept, size=(args.patch_size, args.patch_size), mode='nearest-exact')
    # # percept = torch.nn.functional.interpolate(percept, size=(args.patch_size, args.patch_size), mode='bilinear')
    # print(f'percept shape {percept.shape}')  # shape (N * num_patches, 1, args.patch_size, args.patch_size)
    if args.no_crop:
        percept = pypatchify.unpatchify_from_batches(percept, (percept.shape[1]*16, percept.shape[2]*16, 1), batch_dim=0)  # For debug only
    else:
        percept = pypatchify.unpatchify_from_batches(percept, (args.image_size, args.image_size, 1), batch_dim=0)
    # print(f'percept shape {percept.shape}')  # shape (N, C, args.image_size, args.image_size)
    del patches
    # return percept.expand(-1, 3, -1, -1)
    return np.tile(percept, (1, 1, 1, 3))

def img_generator(files, image_size):
    for image_file in files:
        print(f'{image_file}')
        with open(image_file, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        transform = tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.Grayscale(num_output_channels=3),
            tv_transforms.PILToTensor(),
        ])
        yield image_file, transform(img)

def batch_imgs_generator(files, args):
    # imgs = torch.zeros((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32)
    imgs = np.zeros((args.batch_size, args.image_size, args.image_size, 3))
    img_paths = [None] * args.batch_size
    i = 0
    for image_file in files:
        image_dir, image_name = os.path.split(image_file)
        image_dir = image_dir.split('/')
        image_dir = image_dir[-2:]
        saved_dir = args.output_dir + image_dir[1] + '/'

        """Check if image already exists"""
        if not args.time_track:
            if os.path.isfile(saved_dir + image_name):
                continue

        """Create an output directory if it doesn't exist"""
        Path(saved_dir).mkdir(parents=True, exist_ok=True)

        # print(f'{image_file}')
        # img = read_image(image_file, mode=tv_io_image.ImageReadMode.RGB)
        # with open(image_file, 'rb') as f:
        #     img = Image.open(f)
        #     img = img.convert('RGB')
        # # print(f'img shape {img.shape}')
        # transform = tv_transforms.Compose([
        #     tv_transforms.Resize((args.image_size, args.image_size)),
        #     tv_transforms.ToTensor(),
        # ])
        # img = transform(img)
        # save_image(img, saved_dir + "org_" + image_name)
        img = cv2.imread(image_file)
        img = cv2.resize(img, (args.image_size, args.image_size))
        imgs[i, :, :, :] = np.array(img)

        # print(f'imgs[i] shape {imgs[i].shape}')
        # print(f'imgs[i] type {type(imgs[i])}')

        # save_image(imgs[i], saved_dir + "org_" + image_name)
        # cv2.imwrite(saved_dir + "org_" + image_name, imgs[i]) 
        # image_percept = get_percept(img, p2p_model, square16_implant)
        # image_percept.plot()
        # plt.savefig(saved_dir + '/p2p_img_' + image_name, bbox_inches='tight', pad_inches=0)
        # plt.close()

        img_paths[i] = saved_dir + image_name
        i += 1
        if i == args.batch_size:
            i = 0
            yield img_paths, imgs
    if i > 0:
        yield img_paths, imgs[:len(img_paths), :, :, :]

def main():
    parser = argparse.ArgumentParser('Create percept from pulse2percept')
    parser.add_argument("--image_path", default='/images/PublicDatasets/imagenet/', type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=224, type=int, help="Resize image.")
    parser.add_argument('--patch_size', default=14, type=int, help='Patch resolution of the model.')
    parser.add_argument('--batch_size', default=4096, type=int, help='Batch size.')
    parser.add_argument('--rho', default=150, type=int, help='Rho parameter.')
    parser.add_argument('--axlambda', default=100, type=int, help='Axlambda parameter.')
    parser.add_argument('--range_limit', default=13, type=int, help='xrange min max and yrange min max.')
    parser.add_argument("--range", default=(0, 1), type=int, nargs="+", help="Range of images in dataset.")
    parser.add_argument('--output_dir', default='/images/innoretvision/eye/imagenet_patch/p2p/', help='Path where to save visualizations.')
    parser.add_argument("--no_crop", action='store_true', default=False, help="Crop pulse2percept output.")
    parser.add_argument("--time_track", action='store_true', default=False, help="Track time during the pulse2percept.")
    args = parser.parse_args()

    # args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # args.device = torch.device("cpu")
    
    if args.image_path == '/images/PublicDatasets/imagenet_shared/':
        files = glob.glob(os.path.join(args.image_path, "val/*/*"))  # 50000
        args.output_dir = args.output_dir + r"val" + "/" # + "_G_AP"
    else:
        files = glob.glob(os.path.join(args.image_path, "train/*/*.jpg"))  # 1281167
        args.output_dir = args.output_dir + r"train" + "/" # + "_G_AP"
    # files = glob.glob(os.path.join(args.image_path, "train/*/n02113023_7135.jpg"))
    # random.shuffle(files)
    
    print(f'range {args.range}')
    start = args.range[0]
    end = args.range[1]
    files = files[int(start):int(end)]
    print(f'total files {len(files)}')

    if args.time_track:
        start = time.time()

    # for image_file, img in img_generator(files, args.image_size):
    #     image_dir, image_name = os.path.split(image_file)
    #     image_dir = image_dir.split('/')
    #     image_dir = image_dir[-2:]
    #     saved_dir = args.output_dir + image_dir[1] + '/'

    #     """Check if image already exists"""
    #     img_file = Path(saved_dir + image_name)
    #     if img_file.is_file():
    #         continue

    #     img = torch.tensor(img, device=args.device).unsqueeze(0)
    #     percept = image2percept(img, args)

    #     """Create an output directory if it doesn't exist"""
    #     Path(saved_dir).mkdir(parents=True, exist_ok=True)
    #     save_image(percept[0], saved_dir + image_name)
    
    #     del img, percept, image_name
    #     torch.cuda.empty_cache()
    #     gc.collect()
        
    """Prepare p2p_model and implant"""
    p2p_patch_size = args.patch_size
    p2p_model, square16_implant = build_p2p_model_and_implant(size=p2p_patch_size, axlambda=args.axlambda, rho=args.rho, range_limit=args.range_limit)

    for saved_filenames, imgs in batch_imgs_generator(files, args):
        percept = image2percept(imgs, args, p2p_patch_size, p2p_model, square16_implant)
        for i in range(len(percept)):
            if os.path.isfile(saved_filenames[i]):
                continue
            # save_image(percept[i], saved_filenames[i])
            cv2.imwrite(saved_filenames[i], percept[i]) 
            print(f'{saved_filenames[i]}')
    
        del imgs, percept, saved_filenames
        torch.cuda.empty_cache()
        gc.collect()

    if args.time_track:
        end = time.time()
        print(f"Time elapsed: {end - start}")


if __name__ == '__main__':
    main()
