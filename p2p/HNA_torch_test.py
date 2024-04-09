import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import cv2
import pypatchify

from sklearn.preprocessing import minmax_scale
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem
from HNA_torch import AxonMapSpatialModule, UniversalBiphasicAxonMapModule


p2p_patch_size = 14
SPACE = 300

def build_p2pmodel_and_implant(size, axlambda=100, rho=150, range_limit=13):
    p2pmodel = AxonMapModel(
        axlambda=axlambda, rho=rho, 
        xrange=(-range_limit, range_limit), yrange=(-range_limit, range_limit), xystep=1
    )
    p2pmodel.build()

    implant = ProsthesisSystem(earray=ElectrodeGrid((size, size), SPACE))
    
    return p2pmodel, implant

def get_patient_params(model, targets):
    """Returns patient params (phi) for a patient described by model"""
    return torch.tensor([model.rho, model.axlambda], device=device)
    model_params = torch.tile(torch.tensor([model.rho, model.axlambda], device=device), (len(targets), 1))
    return model_params

device = torch.device("cuda")

start = time.time()

p2pmodel, implant = build_p2pmodel_and_implant(p2p_patch_size, axlambda=1420, rho=437)
decoder = AxonMapSpatialModule(p2pmodel, implant, amp_cutoff=True)
# decoder = UniversalBiphasicAxonMapModule(p2pmodel, implant, amp_cutoff=True)
print(decoder.percept_shape)
decoder.to(device)
for p in decoder.parameters():
    p.requires_grad = False

img = cv2.imread('/home/students/tnguyen/masterthesis/plots/437_1420/train/n03026506/n03026506_2749_stim.jpg', 0)
img = np.expand_dims(np.expand_dims(img, axis=2), axis=0)
print(img.shape, np.max(img), np.min(img))

patches = pypatchify.patchify_to_batches(torch.tensor(img, device=device), (p2p_patch_size, p2p_patch_size, 1), batch_dim=0)
print(patches.shape, torch.max(patches), torch.min(patches))

patches = torch.flatten(patches, start_dim=1)

phis = get_patient_params(p2pmodel, patches)
print(phis.shape)

decoder.eval()
crop_start = (p2p_patch_size // 2) - 1
crop_end = crop_start + p2p_patch_size
percept = torch.zeros((patches.shape[0], 1, p2p_patch_size, p2p_patch_size), device=device)
# for i, patch in enumerate(patches):
#     patch, phis = patch.unsqueeze(0), phis.unsqueeze(0)
#     print(f'patch shape, phis shape {patch.shape, phis.shape}')
#     patch_percept_rescaled = decoder([patch, phis])
#     # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

#     # # take notes of the max of the patch
#     # patch_max = torch.max(patch)
#     # if patch_max > 0.0:
#     #     patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

#     percept[i] = patch_percept_rescaled[:, crop_start:crop_end, crop_start:crop_end]

# percept = decoder([patches, phis])[:, crop_start:crop_end, crop_start:crop_end]

chunk_length = 32
list_of_patches = torch.split(patches, chunk_length)
print(f"chunk shape {list_of_patches[0].shape}")
phis = phis.unsqueeze(0)
for i, patch_chunk in enumerate(list_of_patches):
    print(f'phis shape {phis.shape}')
    patch_percept_rescaled = decoder([patch_chunk, phis])[:, crop_start:crop_end, crop_start:crop_end].unsqueeze(1)
    # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

    # # take notes of the max of the patch
    # patch_max = torch.max(patch_chunk)
    # if patch_max > 0.0:
    #     patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

    percept[i*chunk_length:((i+1)*chunk_length)] = patch_percept_rescaled
    del patch_percept_rescaled

# take notes of the max of the patch
patch_max = torch.max(patches)
if patch_max > 0.0:
    percept = (percept - percept.min())/(percept.max() - percept.min())*patch_max

print(percept.shape)
percept = pypatchify.unpatchify_from_batches(percept, (1, 224, 224), batch_dim=0)

cv2.imwrite('/home/students/tnguyen/masterthesis/plots/437_1420/train/n03026506/AxonMap_torch_test.jpg', np.tile(percept.permute(0, 2, 3, 1).cpu().detach().numpy(), (1, 1, 1, 3)).squeeze())

end = time.time()
print(f"Time elapsed: {end - start}")
