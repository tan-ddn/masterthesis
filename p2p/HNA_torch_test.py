import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import cv2
import pypatchify
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity
from sklearn.preprocessing import minmax_scale
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem, DiskElectrode
from HNA_torch import AxonMapSpatialModule, UniversalBiphasicAxonMapModule
from torch_p2p_dinov2_masked_model import AxonMapSpatialModifiedModule


p2p_patch_size = 14
# rho = 200
# axlambda = 500
# rho = 437
# axlambda = 1420
rho = 150
axlambda = 100
range_limit = 15
xystep = 0.5
disk_electrode_radius = 100
# spacing = 575
spacing = 400

chunk_length = 1

def build_p2pmodel_and_implant(size=14, axlambda=100, rho=150, range_limit=range_limit):
    p2pmodel = AxonMapModel(
        axlambda=axlambda, rho=rho, 
        xrange=(-range_limit, range_limit), yrange=(-range_limit, range_limit), xystep=xystep,
    )
    p2pmodel.build()

    disk_grid = ElectrodeGrid((size, size), spacing, etype=DiskElectrode, r=disk_electrode_radius)  # size = 14  | DiskElectrode?
    # AxonMapModel().plot()
    # disk_grid.plot()
    # plt.savefig('masterthesis/p2p/disk_grid_s400.png')
    # plt.close()

    implant = ProsthesisSystem(earray=disk_grid)
    
    return p2pmodel, implant

def get_patient_params(model, targets):
    """Returns patient params (phi) for a patient described by model"""
    return torch.tensor([model.rho, model.axlambda], device=device)
    model_params = torch.tile(torch.tensor([model.rho, model.axlambda], device=device), (len(targets), 1))
    return model_params

device = torch.device("cuda")

start = time.time()

p2pmodel, implant = build_p2pmodel_and_implant(p2p_patch_size, axlambda=axlambda, rho=rho, range_limit=range_limit)
# # decoder = AxonMapSpatialModule(p2pmodel, implant, amp_cutoff=True)
# decoder = AxonMapSpatialModifiedModule(torch.tensor([[rho]]), torch.tensor([[axlambda]]), p2pmodel, implant, amp_cutoff=True, chunk_length=chunk_length)
# # decoder = UniversalBiphasicAxonMapModule(p2pmodel, implant, amp_cutoff=True)
# print(decoder.percept_shape)
# decoder.to(device)
# for p in decoder.parameters():
#     p.requires_grad = False
# decoder.eval()

decoder2 = AxonMapSpatialModifiedModule(torch.tensor([[rho]]), torch.tensor([[axlambda]]), p2pmodel, implant, amp_cutoff=True, chunk_length=0)
print(decoder2.percept_shape)
decoder2.to(device)
for p in decoder2.parameters():
    p.requires_grad = False
decoder2.eval()

img = cv2.imread('/home/students/tnguyen/masterthesis/plots/437_1420/train/n03026506/n03026506_2749_stim.jpg', 0)
img = np.expand_dims(np.expand_dims(img, axis=2), axis=0)
print(img.shape, np.max(img), np.min(img))

patches = pypatchify.patchify_to_batches(torch.tensor(img, dtype=torch.float, device=device), (p2p_patch_size, p2p_patch_size, 1), batch_dim=0)
# patches /= 255.0
print(patches.shape, torch.max(patches), torch.min(patches))

patches = torch.flatten(patches, start_dim=1)

# phis = get_patient_params(p2pmodel, patches)
# print(phis.shape)

# crop_start = (decoder.percept_shape[0] - p2p_patch_size) // 2
# crop_end = crop_start + p2p_patch_size
margin  = int(decoder2.percept_shape[0] * 0.1)
crop_start = int(decoder2.percept_shape[0] * 0.25) - margin
crop_end = crop_start + int(decoder2.percept_shape[0] * 0.5) + (2 * margin)
crop_size = crop_end - crop_start
percept = torch.zeros((patches.shape[0], crop_size, crop_size), device=device)
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

# before_decoder = time.time()
# chunk_length = chunk_length
# list_of_patches = torch.split(patches, chunk_length)
# print(f"chunk shape {list_of_patches[0].shape}")
# # phis = phis.unsqueeze(0)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
#     with record_function("model_inference"):
#         for i, patch_chunk in enumerate(list_of_patches):  # replace with vectorization
#             # print(f'phis shape {phis.shape}')
#             patch_percept_rescaled = decoder(patch_chunk)[:, crop_start:crop_end, crop_start:crop_end]  # (32, 14x14); (1, 2)
#             # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

#             # # take notes of the max of the patch
#             # patch_max = torch.max(patch_chunk)
#             # if patch_max > 0.0:
#             #     patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

#             percept[i*chunk_length:((i+1)*chunk_length)] = patch_percept_rescaled
#             del patch_percept_rescaled
        
#         # percept = decoder([patches, phis])[:, crop_start:crop_end, crop_start:crop_end]
#         # percept = decoder(patches)[:, crop_start:crop_end, crop_start:crop_end]

#         # take notes of the max of the patch
#         patch_max = torch.max(patches) #* 255
#         if patch_max > 0.0:
#             percept = (percept - percept.min())/(percept.max() - percept.min())*patch_max

#         percept = percept.unsqueeze(1)
#         print(percept.shape)
#         percept = pypatchify.unpatchify_from_batches(percept, (1, 16*p2p_patch_size, 16*p2p_patch_size), batch_dim=0)

# # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# end = time.time()
# print(f"Decoder 1 time elapsed: {end - before_decoder}")

# saving_file_path = f'/home/students/tnguyen/masterthesis/plots/{rho}_{axlambda}/train/n03026506/AxonMap_torch_test_{xystep}.jpg'
# print(saving_file_path)
# cv2.imwrite(saving_file_path, np.tile(percept.permute(0, 2, 3, 1).cpu().detach().numpy(), (1, 1, 1, 3)).squeeze())

before_decoder2 = time.time()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        percept = decoder2(patches)[:, crop_start:crop_end, crop_start:crop_end]

        # take notes of the max of the patch
        patch_max = torch.max(patches) #* 255
        if patch_max > 0.0:
            percept = (percept - percept.min())/(percept.max() - percept.min())*patch_max

        percept = percept.unsqueeze(1)
        print(percept.shape)
        percept = pypatchify.unpatchify_from_batches(percept, (1, 16*crop_size, 16*crop_size), batch_dim=0)

"""Resize percept back to desired shape"""
out = F.interpolate(percept, size=(img.shape[1], img.shape[2]))

# print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
print(f"Decoder 2 time elapsed: {time.time() - before_decoder2}")

# print(f"Time elapsed: {end - start}")  # pytorch profiler

saving_file_path = f'/home/students/tnguyen/masterthesis/plots/{rho}_{axlambda}/train/n03026506/AxonMap_torch_test_resized_{xystep}.jpg'
print(saving_file_path)
cv2.imwrite(saving_file_path, np.tile(out.permute(0, 2, 3, 1).cpu().detach().numpy(), (1, 1, 1, 3)).squeeze())
