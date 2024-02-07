import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pypatchify

from PIL import Image
from pathlib import Path
from numpy import pi
from torchvision import transforms as pth_transforms
from torchvision.utils import save_image
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem
from pulse2percept.stimuli import ImageStimulus

# image_dir = r'/work/scratch/tnguyen/images/cocosearch/patches/'
# images = glob.glob(os.path.join(image_dir, "*/*/*.jpg"))
image_dir = r'/home/students/tnguyen/masterthesis/'
images = glob.glob(os.path.join(image_dir, "n01491361_1000.jpg"))
print(f'Total image: {len(images)}')

IMAGE_SIZE = 224
PATCH_SIZE = 14
SPACE = 300

# index = 0
def get_percept(image, size=(14, 14), axlamda=100, rho=150):
    image_stim = ImageStimulus(image).rgb2gray()

    p2p_model = AxonMapModel(axlambda=axlamda, rho=rho)
    p2p_model.build()

    square16_implant = ProsthesisSystem(earray=ElectrodeGrid(size, SPACE))
    square16_implant.stim = image_stim

    percept = p2p_model.predict_percept(square16_implant)
    return percept

def image2percept(image_path : str = None, image_size : int = IMAGE_SIZE, patch_size : int = PATCH_SIZE):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    image_dir, image_name = os.path.split(image_path)
    image = Image.open(image_path)
    original_image = np.array(image)
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize((image_size, image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    image = transform(image)

    # """Full image to patches"""
    # num_patches = (image_size // patch_size) ** 2
    # image = torch.tensor(image, device=device)
    # patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 2, 3, 0)  # move the channel to the last dim -> (num_patches, w, h, c)
    # patches = patches.cpu().detach().numpy()
    # print(f'patches shape {patches.shape}')
    # # percept = torch.zeros((patches.shape[0], 3, patch_size, patch_size), device=device)
    # percept = torch.zeros((patches.shape[0], 1, patch_size, patch_size), device=device)
    # for i, patch in enumerate(patches):
    #     patch_percept = get_percept(patch, size=(patch_size, patch_size))
    #     patch_percept = torch.tensor(patch_percept, device=device).unsqueeze(0).permute(0, 3, 1, 2)  # move the channel to the second dim
    #     # print(f'patch_percept shape {patch_percept.shape}')
    #     patch_percept = torch.nn.functional.interpolate(patch_percept, size=(patch_size, patch_size), mode='nearest-exact')
    #     print(f'patch_percept shape {patch_percept.shape}')
    #     # percept[i] = patch_percept.expand(-1, 3, -1, -1).squeeze()
    #     percept[i] = patch_percept.squeeze(0)
    # # percept = percept.permute(1, 0, 2, 3).contiguous().view(1, 3, num_patches, -1)
    # percept = percept.contiguous().view(1, 1, num_patches, -1)
    # print(f'percept shape {percept.shape}')  # shape (1, C, num_patches, w * h)
    # percept = percept.permute(0, 1, 3, 2)  # (1, C, w * h, num_patches)
    # # percept = percept.contiguous().view(1, 3*patch_size*patch_size, -1)
    # percept = percept.contiguous().view(1, patch_size*patch_size, -1)
    # print(percept.shape)
    # fold = torch.nn.Fold(output_size=(image_size, image_size), kernel_size=patch_size, stride=patch_size)
    # percept = fold(percept)
    
    """Full image to patches (new method)"""
    num_patches = (image_size // patch_size) ** 2
    image = torch.tensor(image, device=device).unsqueeze(0)#.expand(2, -1, -1, -1)
    print(f'image shape {image.shape}')  # shape (N, C, image_size, image_size)
    patches = pypatchify.patchify_to_batches(image, (patch_size, patch_size), batch_dim=0)
    patches = patches.permute(0, 2, 3, 1).cpu().detach().numpy()
    print(f'patches shape {patches.shape}')  # shape (N * num_patches, patch_size, patch_size, C)
    percept = []
    for i, patch in enumerate(patches):
        patch_percept = get_percept(patch, size=(patch_size, patch_size))
        print(f'patch_percept shape {patch_percept.data.shape}')
        crop = patch_percept.data[30:91, 30:91, :]
        # percept.append(patch_percept.data)
        percept.append(crop)
    percept = torch.tensor(percept, device=device).permute(0, 3, 1, 2)  # move the channel to the second dim
    # print(f'percept shape {percept.shape}')
    percept = torch.nn.functional.interpolate(percept, size=(patch_size, patch_size), mode='nearest-exact')
    print(f'percept shape {percept.shape}')  # shape (N * num_patches, 1, patch_size, patch_size)
    percept = pypatchify.unpatchify_from_batches(percept, (image_size, image_size), batch_dim=0)
    print(f'percept shape {percept.shape}')  # shape (N, C, image_size, image_size)
    

    saved_dir = image_dir + r"/plots"
    """Create an output directory if it doesn't exist"""
    Path(saved_dir).mkdir(parents=True, exist_ok=True)
    print(f'saving image to {saved_dir}')
    image_percept = get_percept(original_image, size=(patch_size, patch_size))
    image_percept.plot()
    plt.savefig(saved_dir + '/p2p_' + str(SPACE) + '_crop_' + image_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    patch_percept.plot()
    plt.savefig(saved_dir + '/p2p_' + str(SPACE) + '_crop_lastpatch_' + image_name, bbox_inches='tight', pad_inches=0)
    plt.close()
    save_image(percept[0], saved_dir + '/p2p_' + str(SPACE) + '_crop_allpatch_' + image_name)

for image in images:
    image2percept(image, IMAGE_SIZE, PATCH_SIZE)
