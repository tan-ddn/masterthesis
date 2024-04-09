import sys
import torch
import torch.nn as nn
import pypatchify
import gc
from torchvision import transforms as pth_transforms
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem

sys.path.append("/home/students/tnguyen/masterthesis")

from p2p.HNA_torch import AxonMapSpatialModule, UniversalBiphasicAxonMapModule


SPACE = 300

def build_p2pmodel_and_implant(size, axlambda=100, rho=150, range_limit=13):
    p2pmodel = AxonMapModel(
        axlambda=axlambda, rho=rho, 
        xrange=(-range_limit, range_limit), yrange=(-range_limit, range_limit), xystep=1
    )
    p2pmodel.build()

    implant = ProsthesisSystem(earray=ElectrodeGrid((size, size), SPACE))
    
    return p2pmodel, implant

def get_patient_params(model, device, targets=None):
    """Returns patient params (phi) for a patient described by model"""
    return torch.tensor([model.rho, model.axlambda], device=device)
    model_params = torch.tile(torch.tensor([model.rho, model.axlambda], device=device), (len(targets), 1))
    return model_params

def image2percept(img, p2p_patch_size, decoder, phis, chunk_length=32):
    crop_start = (p2p_patch_size // 2) - 1
    crop_end = crop_start + p2p_patch_size

    patches = pypatchify.patchify_to_batches(img, (1, p2p_patch_size, p2p_patch_size), batch_dim=0)
    # print(f"patches info {patches.shape}, {patches.max()}, {patches.min()}")

    patches = torch.flatten(patches, start_dim=1)

    percept = torch.zeros((patches.shape[0], 1, p2p_patch_size, p2p_patch_size), device=img.device)
    # for i, patch in enumerate(patches):
    #     patch_percept_rescaled = decoder([patch.unsqueeze(0), phis.unsqueeze(0)])[:, crop_start:crop_end, crop_start:crop_end]
    #     # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

    #     # take notes of the max of the patch
    #     patch_max = torch.max(patch)
    #     if patch_max > 0.0:
    #         patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

    #     percept[i] = patch_percept_rescaled

    # percept = decoder([patches, torch.tile(phis, (len(patches), 1))])[:, crop_start:crop_end, crop_start:crop_end]
    # # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))
    
    chunk_length = chunk_length
    list_of_patches = torch.split(patches, chunk_length)
    # print(f"chunk shape {list_of_patches[0].shape}")
    phis = phis.unsqueeze(0)
    for i, patch_chunk in enumerate(list_of_patches):
        # print(f'phis shape {phis.shape}')
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

    del patches

    # print(f"percept shape {percept.shape}")
    percept = pypatchify.unpatchify_from_batches(percept, (1, 224, 224), batch_dim=0)
    return torch.tile(percept, (1, 3, 1, 1))


class ModelWithMaskedLastFeature(nn.Module):
    def __init__(self, feature_model, top_attention, n_last_blocks, autocast_ctx, args):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.top_attention = top_attention
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
        self.args = args
        self.p2p_patch_size = args.patch_size
        
        p2pmodel, implant = build_p2pmodel_and_implant(self.p2p_patch_size, axlambda=args.axlambda, rho=args.rho)
        self.phis = get_patient_params(p2pmodel, args.device)
        self.decoder = AxonMapSpatialModule(p2pmodel, implant, amp_cutoff=True).to(args.device)
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.to_grayscale_for_p2p = pth_transforms.Grayscale(num_output_channels=1)
        self.percept_norm_for_dino = pth_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                # print(f'top_attention {self.feature_model.top_attention}')
                percepts = image2percept(self.to_grayscale_for_p2p(images), self.p2p_patch_size, self.decoder, self.phis, self.args.chunk_length)
                # print(f"percepts shape, max, and min {percepts.shape, percepts.max(), percepts.min()}")
                features = self.feature_model.get_intermediate_layers_with_masked_feature(
                    self.percept_norm_for_dino(percepts), self.n_last_blocks,
                )
                
        del percepts
        torch.cuda.empty_cache()
        gc.collect()

        return features
