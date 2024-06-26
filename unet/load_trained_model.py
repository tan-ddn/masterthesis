import sys
import torch
import torchvision
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from functools import partial
from torch import nn
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torchvision import transforms as pth_transforms

sys.path.append("/home/students/tnguyen/masterthesis")
sys.path.append("/home/students/tnguyen/masterthesis/dinov2_lib")

from dinov2_lib.dinov2.eval.metrics import MetricType, build_metric
from dinov2_lib.dinov2.eval.linear import _pad_and_collate
from dinov2_lib.dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2_tan.data_transforms import make_classification_eval_transform, make_classification_train_transform
from unet.encoder import ModelWithMaskedLastFeature, EncoderModel, Pipeline
from unet.train_encoder import get_args_parser, setup_and_build_model, setup_linear_classifiers, ImageNetWds
from p2p.torch_p2p_dinov2_masked_model import image2percept


PATH = r"/work/scratch/tnguyen/unet/encoder/16/model_final.pth"
UNET2_A_PATH = r"/work/scratch/tnguyen/unet/encoder/23/model_0002939.pth"
# UNET2_A_PATH = r"/work/scratch/tnguyen/unet/encoder/23/running_checkpoint_linear_eval.pth"
# UNET2_A_PATH = r"/work/scratch/tnguyen/unet/encoder/23/model_0000881.pth"
LINEAR_END_A_PATH = r"/work/scratch/tnguyen/unet/encoder/17/model_0002939.pth"
LINEAR_BEGIN_A_PATH = r"/work/scratch/tnguyen/unet/encoder/17/model_0000587.pth"

UNET2_A_EPOCH1_PATH = r"/work/scratch/tnguyen/unet/encoder/39/model_0000146.pth"

# LINEAR_A_PATH_1 = r"/work/scratch/tnguyen/unet/encoder/42/model_0000294.pth"
# LINEAR_A_PATH_2 = r"/work/scratch/tnguyen/unet/encoder/42/model_0001769.pth"
# LINEAR_A_PATH_3 = r"/work/scratch/tnguyen/unet/encoder/42/running_checkpoint_linear_eval.pth"

LINEAR_A_PATH_1 = r"/work/scratch/tnguyen/unet/encoder/41/model_0000072.pth"
LINEAR_A_PATH_2 = r"/work/scratch/tnguyen/unet/encoder/41/model_0000656.pth"
LINEAR_A_PATH_3 = r"/work/scratch/tnguyen/unet/encoder/41/running_checkpoint_linear_eval.pth"

def make_eval_data_loader(test_dataset_str, num_workers, metric_type):
    resize_size = int(224 * 1.15)
    if test_dataset_str == 'ImageNetWds':
        test_transform = make_classification_eval_transform(resize_size=resize_size, crop_size=224, grayscale=False, norm="no_norm")
        test_data_dir = r'/images/innoretvision/eye/imagenet_patch/val/'
        test_data_num = r'006'
        test_data_path = test_data_dir + 'imagenet-val-{000000..000' + test_data_num + '}.tar'
        pil_dataset = (
            ImageNetWds(
            # wids.ShardListDataset(
                test_data_path,
            )
            .set_split('val')
            # .shuffle(5000)
            .decode("pil")
            .to_tuple("jpg", "cls")
        )

        def preprocess(sample):
            image, label = sample
            # image, label = sample[".jpg"], sample[".cls"]
            return test_transform(image), label

        test_dataset = pil_dataset.map(preprocess)
        
        test_data_loader = make_data_loader(
            dataset=test_dataset,
            batch_size=2,
            num_workers=num_workers,
            sampler_type=None,
            drop_last=False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
        )
    else:
        if test_dataset_str == 'Imagenette':
            test_dataset = torchvision.datasets.ImageFolder(
                root=r"/work/scratch/tnguyen/images/imagenette2/val",
                transform=make_classification_eval_transform(resize_size=resize_size, crop_size=224, grayscale=False, norm="no_norm"),
                # transform=make_classification_train_transform(crop_size=224, grayscale=False, norm="no_norm"),
            )
        else:
            test_dataset = make_dataset(
                dataset_str=test_dataset_str,
                transform=make_classification_eval_transform(resize_size=resize_size, crop_size=224, grayscale=False, norm="no_norm"),
            )
        test_data_loader = make_data_loader(
            dataset=test_dataset,
            batch_size=2,
            num_workers=num_workers,
            sampler_type=SamplerType.DISTRIBUTED,
            drop_last=False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
        )
    return test_data_loader, test_dataset

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    description = "DINOv2 Visualize Self-Attention maps"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

    args.device = device
    args.config_file = r"/home/students/tnguyen/masterthesis/dinov2_lib/dinov2/configs/eval/vits14_pretrain.yaml"
    args.pretrained_weights = r"/home/students/tnguyen/masterthesis/dinov2_lib/dinov2_vits14_pretrain.pth"
    args.torch_p2p = True
    args.learning_rates = [0.02]

    # ckpt = torch.load(UNET2_A_PATH)
    ckpt = torch.load(UNET2_A_EPOCH1_PATH)
    # ckpt = torch.load(LINEAR_BEGIN_A_PATH)
    # ckpt = torch.load(LINEAR_END_A_PATH)
    # ckpt = torch.load(LINEAR_A_PATH_1)
    # ckpt = torch.load(LINEAR_A_PATH_2)
    # ckpt = torch.load(LINEAR_A_PATH_3)
    
    # for key, value in ckpt.items():
        # if key not in ["optimizer", "scheduler", "iteration"]:
        #     print(f"key {key}")
    # model = ckpt["model"]
    # for key, value in model.items():
    #     print(key)
    # sys.exit(0)

    val_data_loader, val_dataset = make_eval_data_loader("ImageNetWds", 1, MetricType.MEAN_ACCURACY)

    dinov2_model, autocast_dtype = setup_and_build_model(args)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    encoder_model = EncoderModel(patch_size=14, n_channels=1, n_classes=1, bilinear=False, encoder=args.encoder)
    masked_torch_p2p_dinov2_model = ModelWithMaskedLastFeature(dinov2_model, 0.1, 4, autocast_ctx, args)
    feature_model = Pipeline(encoder=encoder_model, torch_p2p_dinov2=masked_torch_p2p_dinov2_model)
    feature_model = feature_model.to(device=args.device)

    for image, label in val_dataset:
        break
    print(f"Image shape, max, and min {image.shape, image.max(), image.min()}")
    sample_output = feature_model(image.unsqueeze(0).cuda())
    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        [1, 4],
        # fixations,
        args.learning_rates,
        2,
        10,
    )
    checkpoint_model = nn.Sequential(feature_model.encoder, linear_classifiers)
    checkpoint_model.load_state_dict(ckpt['model'])
    checkpoint_model.eval()
    # for p in checkpoint_model.parameters():
    #     p.requires_grad = False
    # encoder_model.load_state_dict(ckpt['model'].items()[0])
    # encoder_model.eval()
    for image, label in val_dataset:
        break
    # torchvision.utils.save_image(image, r"/home/students/tnguyen/masterthesis/unet/original_image.jpg")
    # image = image.unsqueeze(0).cuda()
    # output = checkpoint_model[0](image)
    # # output = encoder_model(image)
    # torchvision.utils.save_image(output[0], r"/home/students/tnguyen/masterthesis/unet/after_unet2_image.jpg")
    
    # to_grayscale_for_p2p = pth_transforms.Grayscale(num_output_channels=1)
    # percepts = image2percept(to_grayscale_for_p2p(image), 14, masked_torch_p2p_dinov2_model.decoder, 2)
    # torchvision.utils.save_image(percepts[0], r"/home/students/tnguyen/masterthesis/unet/original_after_p2p_image.jpg")
    # percepts = image2percept(output, 14, masked_torch_p2p_dinov2_model.decoder, 2)
    # torchvision.utils.save_image(percepts[0], r"/home/students/tnguyen/masterthesis/unet/after_unet2_p2p_image.jpg")


    """Draw histogram of the trained weights"""
    nbins = 19
    trained_weights = torch.tensor([], device=args.device)
    for name, p in checkpoint_model[0].unet2.named_parameters():
        print(f"{name}: {p.requires_grad}, {p.grad}")
        p.requires_grad = False
        p_1d = p.view(p.nelement())
        trained_weights = torch.cat((trained_weights, p_1d))
    print(f"trained weights shape {trained_weights.shape}")
    print(f"max min mean std: {torch.max(trained_weights), torch.min(trained_weights), torch.mean(trained_weights), torch.std(trained_weights)}")

    # trained_hist = numpy.histogram(trained_weights.cpu().detach().numpy(), bins=nbins, range=(-1., 1.), density=False)
    # # print(trained_hist)
    # hist_values, bin_edges = trained_hist[0].astype(numpy.int64), trained_hist[1]
    # hist_values = hist_values / hist_values.sum()
    # # print(hist_values)
    # # print(hist_values.sum())
    # x_range = (bin_edges[1:] + bin_edges[:-1]) / 2
    # width = (x_range[1] - x_range[0]) * 0.8
    # fig, ax = plt.subplots()
    # ax.bar(x_range, hist_values, width=width, align='center')
    # # ax.hist(hist_values, bins=bin_edges, align='mid', density=False)
    # # ax.set_xlabel('Bin range')
    # # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # plt.savefig(r"/home/students/tnguyen/masterthesis/unet/after_trained_unet2_weights.png")
    # plt.close()


    # """Check weights of linear classifiers"""
    # trained_weights = torch.tensor([], device=args.device)
    # for name, param in checkpoint_model[1].named_parameters():
    #     # print(name) #, param.data
    #     param.requires_grad = False
    #     param_1d = param.view(param.nelement())
    #     trained_weights = torch.cat((trained_weights, param_1d))
    # print(f"trained weights shape {trained_weights.shape}")
    # print(f"max min mean std: {torch.max(trained_weights), torch.min(trained_weights), torch.mean(trained_weights), torch.std(trained_weights)}")


    # """Draw histogram of the linear weights"""
    # nbins = 19
    # trained_weights = torch.tensor([], device=args.device)
    # for name, p in checkpoint_model[0].linear.named_parameters():
    #     print(name)
    #     p.requires_grad = False
    #     p_1d = p.view(p.nelement())
    #     trained_weights = torch.cat((trained_weights, p_1d))
    # print(f"trained weights shape {trained_weights.shape}")
    # print(f"max min mean std: {torch.max(trained_weights), torch.min(trained_weights), torch.mean(trained_weights), torch.std(trained_weights)}")

    # trained_hist = numpy.histogram(trained_weights.cpu().detach().numpy(), bins=nbins, range=(-1., 1.), density=False)
    # # print(trained_hist)
    # hist_values, bin_edges = trained_hist[0].astype(numpy.int64), trained_hist[1]
    # hist_values = hist_values / hist_values.sum()
    # # print(hist_values)
    # # print(hist_values.sum())
    # x_range = (bin_edges[1:] + bin_edges[:-1]) / 2
    # width = (x_range[1] - x_range[0]) * 0.8
    # fig, ax = plt.subplots()
    # ax.bar(x_range, hist_values, width=width, align='center')
    # # ax.hist(hist_values, bins=bin_edges, align='mid', density=False)
    # # ax.set_xlabel('Bin range')
    # # ax.set_yscale('log')
    # ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    # plt.savefig(r"/home/students/tnguyen/masterthesis/unet/after_trained_unet2_weights.png")
    # plt.close()
