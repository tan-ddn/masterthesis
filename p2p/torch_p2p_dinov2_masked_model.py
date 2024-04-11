import sys
import torch
import torch.nn as nn
import pypatchify
import gc
from torchvision import transforms as pth_transforms
from pulse2percept.models import AxonMapModel
from pulse2percept.implants import ElectrodeGrid, ProsthesisSystem, DiskElectrode

sys.path.append("/home/students/tnguyen/masterthesis")

from p2p.HNA_torch import AxonMapSpatialModule, UniversalBiphasicAxonMapModule, calc_axon_sensitivity


range_limit = 5
xystep = 0.75
disk_electrode_radius = 100
spacing = 575

def build_p2pmodel_and_implant(size=14, axlambda=100, rho=150, range_limit=range_limit):
    p2pmodel = AxonMapModel(
        axlambda=axlambda, rho=rho, 
        xrange=(-range_limit, range_limit), yrange=(-range_limit, range_limit), xystep=xystep
    )
    p2pmodel.build()

    disk_grid = ElectrodeGrid((size, size), spacing, etype=DiskElectrode, r=disk_electrode_radius)  # size = 14  | DiskElectrode?
    implant = ProsthesisSystem(earray=disk_grid)
    
    return p2pmodel, implant

def get_patient_params(model, device, targets=None):
    """Returns patient params (phi) for a patient described by model"""
    return torch.tensor([model.rho, model.axlambda], device=device)
    model_params = torch.tile(torch.tensor([model.rho, model.axlambda], device=device), (len(targets), 1))
    return model_params

def image2percept(img, p2p_patch_size, decoder, chunk_length=64):
    crop_start = (decoder.percept_shape[0] - p2p_patch_size) // 2
    crop_end = crop_start + p2p_patch_size

    patches = pypatchify.patchify_to_batches(img, (1, p2p_patch_size, p2p_patch_size), batch_dim=0) # 512 images x 256 patches/img 
    # print(f"patches info {patches.shape}, {patches.max()}, {patches.min()}")

    patches = torch.flatten(patches, start_dim=1)

    percept = torch.zeros((patches.shape[0], p2p_patch_size, p2p_patch_size), device=img.device)
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
    
    if chunk_length > 0:
        chunk_length = chunk_length
        list_of_patches = torch.split(patches, chunk_length)
        # print(f"chunk shape {list_of_patches[0].shape}")
        for i, patch_chunk in enumerate(list_of_patches):
            # print(f'phis shape {phis.shape}')
            patch_percept_rescaled = decoder(patch_chunk)[:, crop_start:crop_end, crop_start:crop_end]
            # print(patch_percept_rescaled.shape, torch.max(patch_percept_rescaled), torch.min(patch_percept_rescaled))

            # # take notes of the max of the patch
            # patch_max = torch.max(patch_chunk)
            # if patch_max > 0.0:
            #     patch_percept_rescaled = (patch_percept_rescaled - patch_percept_rescaled.min())/(patch_percept_rescaled.max() - patch_percept_rescaled.min())*patch_max

            percept[i*chunk_length:((i+1)*chunk_length)] = patch_percept_rescaled
            del patch_percept_rescaled
    else:
        percept = decoder(patches)[:, crop_start:crop_end, crop_start:crop_end]

    # take notes of the max of the patch
    patch_max = torch.max(patches)
    if patch_max > 0.0:
        percept = (percept - percept.min())/(percept.max() - percept.min())*patch_max

    del patches
    torch.cuda.empty_cache()
    gc.collect()

    percept = percept.unsqueeze(1)
    # print(f"percept shape {percept.shape}")
    percept = pypatchify.unpatchify_from_batches(percept, (1, 224, 224), batch_dim=0) # 512 images
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
        # self.phis = get_patient_params(p2pmodel, args.device)
        # self.decoder = AxonMapSpatialModule(p2pmodel, implant, amp_cutoff=True).to(args.device)
        self.decoder = AxonMapSpatialModifiedModule(torch.tensor([[args.rho]]), torch.tensor([[args.axlambda]]), p2pmodel, implant, amp_cutoff=True, chunk_length=self.args.chunk_length).to(args.device)
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        self.to_grayscale_for_p2p = pth_transforms.Grayscale(num_output_channels=1)
        self.percept_norm_for_dino = pth_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                # print(f'top_attention {self.feature_model.top_attention}')
                percepts = image2percept(self.to_grayscale_for_p2p(images), self.p2p_patch_size, self.decoder, self.args.chunk_length)
                # print(f"percepts shape, max, and min {percepts.shape, percepts.max(), percepts.min()}")
                features = self.feature_model.get_intermediate_layers_with_masked_feature(
                    self.percept_norm_for_dino(percepts), self.n_last_blocks,
                )
                
        del percepts
        torch.cuda.empty_cache()
        gc.collect()

        return features


class AxonMapSpatialModifiedModule(torch.nn.Module):
    def __init__(self, rho, axlambda, p2pmodel, implant, activity_regularizer=None, clip=None, amp_cutoff=True, chunk_length=0, **kwargs):
        super().__init__()

        self.alterntive_fwd = True if chunk_length == 0 else False

        dtype = torch.get_default_dtype()

        # p2pmodel.min_ax_sensitivity = 0.2
        bundles = p2pmodel.grow_axon_bundles() # 763 [[20-300]x,y]
        # ok beyeler2019

        axons = p2pmodel.find_closest_axon(bundles) # 2401 [[20-300]x,y]
        # ok beyeler2019

        if type(axons) != list:
            axons = [axons]
        axon_contrib = calc_axon_sensitivity(p2pmodel, axons, pad=True) # similar beyeler2019 without axlambda
        axon_contrib = torch.tensor(axon_contrib, dtype=dtype) # 2401 pix, 118 l_ax, 3 (x,y,sens)

        # Get implant parameters
        # self.n_elecs = len(implant.electrodes)
        self.elec_x = torch.tensor([implant[e].x for e in implant.electrodes], dtype=dtype)
        self.elec_y = torch.tensor([implant[e].y for e in implant.electrodes], dtype=dtype)

        d2_el = (axon_contrib[:, :, 0, None] - self.elec_x)**2 + \
                (axon_contrib[:, :, 1, None] - self.elec_y)**2 # 2401, 118, 225

        self.clip = False
        if isinstance(clip, tuple):
            self.clip = True
            self.clipmin = clip[0]
            self.clipmax = clip[1]

        self.amp_cutoff = amp_cutoff
        self.percept_shape = p2pmodel.grid.shape
        self.thresh_percept = p2pmodel.thresh_percept
        
        p1 = -d2_el[None, :, :, :]
        p2 = (2. * rho**2)[:, None, None, :]
        p3 = axon_contrib[None, :, :, 2, None]
        p4 = (axlambda**2)[:, None, None, :]
        # print(f'p shape {p1.shape, p2.shape, p3.shape, p4.shape}')
        p_exp = torch.exp(   # gauss
                                    p1 / # dist2el 1, 2401, 118, 225
                                    p2 # 1, 1, 1, 225
                                    + # contribution of each electode to each axon segement of each
                                      # pixel by distance of segemnt to electrode
                                    p3 / # sens 1, 2401, 118, 1
                                    p4 # 1, 1, 1 , 225
                                      # contribution of each electode to each axon segement of each
                                      # pixel by sensitivity, which is scaled by distance along axon
                                 ) # 1, 2401, 118, 225, scaling between 0, 1
        print(f'p_exp shape {p_exp.shape}')
        self.register_buffer("p_exp", p_exp)

    def forward(self, amp):
        # print(f'p_exp device {self.p_exp.device}')
        if not self.alterntive_fwd:
            # apply axon map
            intensities =   (
                            amp[:, None, None, :] * # b, 1, 1, 196
                            self.p_exp # 1, 729, 191, 196
                            ) # b, 729, 191, 196
            # print(f'intensities shape {intensities.shape}')

            # after summing up...
            intensities_per_axon = torch.sum(intensities, axis=-1)
        else:
            """Alternative method"""
            amp = amp[:, None, None, :]
            intensities_per_axon = ([(a_ * self.p_exp).sum(axis=-1) for a_ in amp])
            intensities_per_axon = torch.stack(intensities_per_axon, dim=0)
        
        intensities = torch.take_along_dim(
            intensities_per_axon, intensities_per_axon.abs().max(-1, keepdim=True).indices, dim=-1).squeeze(-1)
        del intensities_per_axon

        intensities = torch.where(intensities.abs() > self.thresh_percept, intensities, torch.zeros_like(intensities))

        if self.clip:
            intensities = torch.clamp(intensities, self.clipmin, self.clipmax)

        batched_percept_shape = tuple([-1] + list(self.percept_shape))
        intensities = intensities.reshape(batched_percept_shape)
        return intensities
