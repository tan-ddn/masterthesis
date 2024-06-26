import sys
import torch
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

sys.path.append("/home/students/tnguyen/masterthesis")
sys.path.append("/home/students/tnguyen/masterthesis/dinov2_lib")

from unet.encoder import ModelWithMaskedLastFeature, EncoderModel, Pipeline


numpy.set_printoptions(threshold=sys.maxsize)
# PATH = r"/work/scratch/tnguyen/unet/encoder/16/model_final.pth"
UNET2_I_PATH = r"/work/scratch/tnguyen/unet/identity_encoder/3/checkpoint_epoch10.pth"

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    ckpt = torch.load(UNET2_I_PATH)
    # for key, value in ckpt.items():
    #     print(f"key {key}")
    # sys.exit(0)

    encoder_model = EncoderModel(patch_size=14, n_channels=1, n_classes=1, bilinear=False, 
                                 encoder="unet2", down_sampling=False)
    
    nbins = 19

    """Draw histogram of the initial weights"""
    initial_weights = torch.tensor([])
    for p in encoder_model.unet2.parameters():
        p.requires_grad = False
        # print(p.shape, p.nelement())
        p_1d = p.view(p.nelement())
        # print(p_1d.shape)
        initial_weights = torch.cat((initial_weights, p_1d))
    print(f"initial weights shape {initial_weights.shape}")
    initial_hist = numpy.histogram(initial_weights.detach().numpy(), bins=nbins, range=(-1., 1.), density=False)
    print(initial_hist)
    hist_values, bin_edges = initial_hist[0].astype(numpy.int64), initial_hist[1]
    hist_values = hist_values / hist_values.sum()
    x_range = (bin_edges[1:] + bin_edges[:-1]) / 2
    width = (x_range[1] - x_range[0]) * 0.8
    fig, ax = plt.subplots()
    ax.bar(x_range, hist_values, width=width, align='center')
    # ax.hist(hist_values, bins=bin_edges, align='mid', density=False)
    # ax.set_xlabel('Bin range')
    # ax.set_yscale('log')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.savefig(r"/home/students/tnguyen/masterthesis/unet/initial_unet2_weights.png")
    plt.close()
    
    """Draw histogram of the identity weights"""
    encoder_model.unet2.load_state_dict(ckpt)
    encoder_model.unet2.eval()
    identity_weights = torch.tensor([])
    for p in encoder_model.unet2.parameters():
        p.requires_grad = False
        p_1d = p.view(p.nelement())
        identity_weights = torch.cat((identity_weights, p_1d))
    print(f"identity weights shape {identity_weights.shape}")
    print(f"max min mean std: {torch.max(identity_weights), torch.min(identity_weights), torch.mean(identity_weights), torch.std(identity_weights)}")
    identity_hist = numpy.histogram(identity_weights.detach().numpy(), bins=nbins, range=(-1., 1.), density=False)
    print(identity_hist)
    hist_values, bin_edges = identity_hist[0].astype(numpy.int64), identity_hist[1]
    hist_values = hist_values / hist_values.sum()
    print(hist_values)
    print(hist_values.sum())
    x_range = (bin_edges[1:] + bin_edges[:-1]) / 2
    width = (x_range[1] - x_range[0]) * 0.8
    fig, ax = plt.subplots()
    ax.bar(x_range, hist_values, width=width, align='center')
    # ax.hist(hist_values, bins=bin_edges, align='mid', density=False)
    # ax.set_xlabel('Bin range')
    # ax.set_yscale('log')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    plt.savefig(r"/home/students/tnguyen/masterthesis/unet/identity_unet2_weights.png")
    plt.close()
