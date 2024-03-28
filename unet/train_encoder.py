import argparse
import logging
import os
import glob
import cv2
import numpy as np
import random
import sys
import json
import pypatchify
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tv_transforms
from torchvision.utils import save_image
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
from random import shuffle
from torchsummary import summary


sys.path.append("/home/students/tnguyen/masterthesis")

from unet.unet_model import UNet
from unet.evaluate import evaluate
from dinov2_tan.attn_eval_linear import *


class EncoderModel(UNet):
    def __init__(self, patch_size, n_channels, n_classes, bilinear=False):
        super(EncoderModel, self).__init__(n_channels, n_classes, bilinear)
        self.patch_size = patch_size
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = pypatchify.patchify_to_batches(x, (3, self.patch_size, self.patch_size), batch_dim=0)
        # print(f'x shape {x.shape}')
        x = super(EncoderModel, self).forward(x)
        x = pypatchify.unpatchify_from_batches(x, (C, H, W), batch_dim=0)
        return x

class Pipeline(nn.Module):
    def __init__(self, encoder, dinov2):
        super(Pipeline, self).__init__()
        self.encoder = encoder
        self.dinov2 = dinov2
        
    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.dinov2(x1)
        return x2

def get_args(parents = None):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(description='Train the UNet on images and target', parents=parents, )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--test-datasets",
        dest="test_dataset_strs",
        type=str,
        nargs="+",
        help="Test datasets, none to reuse the validation dataset",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number de Workers",
    )
    parser.add_argument(
        "--epoch-length",
        type=int,
        help="Length of an epoch in number of iterations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=float,
        help="Number of epochs between two named checkpoint saves.",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between two evaluations.",
    )
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        help="Learning rates to grid search.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not resume from existing checkpoints",
    )
    parser.add_argument(
        "--val-metric-type",
        type=MetricType,
        choices=list(MetricType),
        help="Validation metric",
    )
    parser.add_argument(
        "--test-metric-types",
        type=MetricType,
        choices=list(MetricType),
        nargs="+",
        help="Evaluation metric",
    )
    parser.add_argument(
        "--classifier-fpath",
        type=str,
        help="Path to a file containing pretrained linear classifiers",
    )
    parser.add_argument(
        "--val-class-mapping-fpath",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument(
        "--test-class-mapping-fpaths",
        nargs="+",
        type=str,
        help="Path to a file containing a mapping to adjust classifier outputs",
    )
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--fixation_grayscale", action="store_true", help="Fixations in grayscale or rgb. Default is rgb.")
    parser.add_argument("--fixation_top", default=0.1, type=float, help="Percentage of fixations with top attention score.")
    parser.add_argument("--image_grayscale", action="store_true", help="Image in grayscale or rgb. Default is rgb.")
    parser.add_argument("--n_last_blocks", default=4, type=int, help="Maximun number of last blocks used for linear probing.")
    parser.add_argument("--pulse2percept", action="store_true", help="Pulse2percept between dataset and dinov2.")
    parser.add_argument("--p2p_no_tar", action="store_true", help="Use pulse2percept tar files or not.")
    parser.add_argument("--norm", default="norm", type=str,
        help='Normalization method for images: "norm", "no_norm", "norm_after_p2p". "norm_after_p2p" is only used when pulse2percept and p2p_no_tar are both True. Default: "norm"')
    
    parser.add_argument('--image_size', metavar='I', type=int, default=224, help='Full image size')
    parser.add_argument('--patch_size', metavar='P', type=int, default=14, help='Patch size')
    parser.add_argument('--epochs', metavar='E', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-w', metavar='WD', type=float, default=0, help='Weight decay', dest='wd')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load unet model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument("--run_on_cluster", action="store_true", help="Run on cluster or local machine. Default: local machine.")

    parser.set_defaults(
        train_dataset_str="ImageNetWds",
        val_dataset_str="ImageNetWds",
        test_dataset_strs=None,
        epochs=10,
        batch_size=128,
        num_workers=1,
        epoch_length=10001,
        save_checkpoint_frequency=1,
        eval_period_iterations=10001,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
        patch_size=14,
        image_size=224,
        checkpoint_key="teacher",
        fixation_grayscale=False,
        fixation_top=0.1,
        image_grayscale=False,
        n_last_blocks=4,
        run_on_cluster=False,
        pulse2percept=False,
        p2p_no_tar=False,
        norm='norm',
        output_dir='/work/scratch/tnguyen/unet/encoder/0/',
    )

    return parser.parse_args()

def run_pipeline(
    encoder_model,
    dinov2_model,
    output_dir,
    train_dataset_str,
    val_dataset_str,
    batch_size,
    epochs,
    epoch_length,
    num_workers,
    save_checkpoint_frequency,
    eval_period_iterations,
    learning_rates,
    autocast_dtype,
    test_dataset_strs=None,
    resume=True,
    classifier_fpath=None,
    val_class_mapping_fpath=None,
    test_class_mapping_fpaths=[None],
    val_metric_type=MetricType.MEAN_ACCURACY,
    test_metric_types=None,
):
    seed = 0

    if test_dataset_strs is None:
        test_dataset_strs = [val_dataset_str]
    if test_metric_types is None:
        test_metric_types = [val_metric_type] * len(test_dataset_strs)
    else:
        assert len(test_metric_types) == len(test_dataset_strs)
    assert len(test_dataset_strs) == len(test_class_mapping_fpaths)

    p2p_tar_transform = True if args.pulse2percept and (not args.p2p_no_tar) else False
    train_transform = make_classification_train_transform(crop_size=args.image_size, grayscale=args.image_grayscale, norm=args.norm, pulse2percept=p2p_tar_transform)
    if train_dataset_str == 'ImageNetWds':
        train_data_dir = r'/images/innoretvision/eye/imagenet_patch/train/'
        train_data_num = r'020'
        if args.run_on_cluster:
            train_data_num = r'146'
        if p2p_tar_transform:
            train_data_dir = r'/images/innoretvision/eye/imagenet_patch/p2p/train_shards/'
            train_data_num = r'008'
            if args.run_on_cluster:
                train_data_num = r'027'
        train_data_path = train_data_dir + 'imagenet-train-{000000..000' + train_data_num + '}.tar'
        pil_dataset = (
            ImageNetWds(
            # wids.ShardListDataset(
                train_data_path,
            )
            .shuffle(5000)
            .decode("pil")
            .to_tuple("jpg", "cls")
        )

        def preprocess(sample):
            image, label = sample
            # image, label = sample[".jpg"], sample[".cls"]
            return train_transform(image), label

        train_dataset = pil_dataset.map(preprocess)
        training_num_classes = 1000
        sampler_type = None
    else:
        train_dataset = make_dataset(
            dataset_str=train_dataset_str,
            transform=train_transform,
        )
        training_num_classes = len(torch.unique(torch.Tensor(train_dataset.get_targets().astype(int))))
        sampler_type = SamplerType.SHARDED_INFINITE
        # sampler_type = SamplerType.INFINITE

    # n_last_blocks_list = [1, 4]
    # n_last_blocks = max(n_last_blocks_list)
        
    n_last_blocks_list = [1]
    n_last_blocks = max(n_last_blocks_list)
    if args.n_last_blocks > 1:
        n_last_blocks_list = [1, args.n_last_blocks]
        n_last_blocks = max(n_last_blocks_list)

    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    fixations = None
    # fixation_model = ModelLastSelfAttentionFixation(model, args, 2, autocast_ctx).to(args.device)
    if args.fixation_top >= 1:
        masked_dinov2_model = ModelWithIntermediateLayers(dinov2_model, n_last_blocks, autocast_ctx)
        masked_compared_model = ModelWithMaskedLastFeature(dinov2_model, args.fixation_top, n_last_blocks, autocast_ctx, args)
    else:
        masked_dinov2_model = ModelWithMaskedLastFeature(dinov2_model, args.fixation_top, n_last_blocks, autocast_ctx, args)
    feature_model = Pipeline(encoder=encoder_model, dinov2=masked_dinov2_model)
    logger.info(summary(Pipeline(encoder=encoder_model, dinov2=dinov2_model), (3, args.image_size, args.image_size)))
    if train_dataset_str == 'ImageNetWds':
        for image, label in train_dataset:
            break
        # sample_output, fixations = feature_model(image.unsqueeze(0).cuda())
        # _, fixations = fixation_model(image.unsqueeze(0).cuda())
        # sample_output = feature_model(fixations)
        sample_output = feature_model(image.unsqueeze(0).cuda())
        # if args.fixation_top >= 1:
        #     compared_sample_output = compared_model(image.unsqueeze(0).cuda())
        #     mat_A = sample_output[0][0]#, sample_output[0][1]
        #     mat_B = compared_sample_output[0][0]#, compared_sample_output[0][1]
        #     print(mat_A, mat_B)
        #     print(mat_A.shape, mat_B.shape)
        #     print(f"compare 2 models' outputs: {torch.allclose(mat_A, mat_B)}")
        #     print(f"similar percentage: {torch.sum(torch.eq(mat_A, mat_B)).item()/mat_A.nelement()}")
        #     exit()
    else:
        # sample_output, fixations = feature_model(train_dataset[0][0].unsqueeze(0).cuda())
        # _, fixations = fixation_model(train_dataset[0][0].unsqueeze(0).cuda())
        # sample_output = feature_model(fixations)
        sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        n_last_blocks_list,
        # fixations,
        learning_rates,
        batch_size,
        training_num_classes,
    )
    del fixations, sample_output

    optimizer = torch.optim.SGD(optim_param_groups, momentum=0.9, weight_decay=0)
    max_iter = epochs * epoch_length
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1
    train_data_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        sampler_type=sampler_type,
        sampler_advance=start_iter,
        drop_last=True,
        persistent_workers=True,
    )
    val_data_loader = make_eval_data_loader(val_dataset_str, args, num_workers, val_metric_type)

    checkpoint_period = int(save_checkpoint_frequency * epoch_length)

    if val_class_mapping_fpath is not None:
        logger.info(f"Using class mapping from {val_class_mapping_fpath}")
        val_class_mapping = np.load(val_class_mapping_fpath)
    else:
        val_class_mapping = None

    test_class_mappings = []
    for class_mapping_fpath in test_class_mapping_fpaths:
        if class_mapping_fpath is not None and class_mapping_fpath != "None":
            logger.info(f"Using class mapping from {class_mapping_fpath}")
            class_mapping = np.load(class_mapping_fpath)
        else:
            class_mapping = None
        test_class_mappings.append(class_mapping)

    metrics_file_path = os.path.join(output_dir, "results_eval_linear.json")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}")
        val_results_dict, feature_model, linear_classifiers, iteration = eval_linear(
            # fixation_model=fixation_model,
            feature_model=feature_model,
            linear_classifiers=linear_classifiers,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            metrics_file_path=metrics_file_path,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            max_iter=max_iter,
            checkpoint_period=checkpoint_period,
            running_checkpoint_period=epoch_length,
            eval_period=eval_period_iterations,
            metric_type=val_metric_type,
            training_num_classes=training_num_classes,
            resume=resume,
            val_class_mapping=val_class_mapping,
            classifier_fpath=classifier_fpath,
        )
        gc.collect()
        ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        print(f"ram usage: {ram_usage}")
    results_dict = {}
    if len(test_dataset_strs) > 1 or test_dataset_strs[0] != val_dataset_str:
        results_dict = test_on_datasets(
            # fixation_model,
            feature_model,
            linear_classifiers,
            test_dataset_strs,
            batch_size,
            0,  # num_workers,
            test_metric_types,
            metrics_file_path,
            training_num_classes,
            iteration,
            val_results_dict["best_classifier"]["name"],
            prefixstring="",
            test_class_mappings=test_class_mappings,
        )
    results_dict["best_classifier"] = val_results_dict["best_classifier"]["name"]
    results_dict[f"{val_dataset_str}_accuracy"] = 100.0 * val_results_dict["best_classifier"]["accuracy"]
    logger.info("Test Results Dict " + str(results_dict))

    return results_dict


if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    args.device = device

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    encoder_model = EncoderModel(patch_size=args.patch_size, n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    encoder_model = encoder_model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{encoder_model.n_channels} input channels\n'
                 f'\t{encoder_model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if encoder_model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        encoder_model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    encoder_model.to(device=device)
    
    """Dinov2 part"""
    dinov2_model, autocast_dtype = setup_and_build_model(args)

    for p in dinov2_model.parameters():
        p.requires_grad = False
    dinov2_model.eval()
    dinov2_model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = dinov2_model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    else:
        print("Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate.")
        url = None
        if args.arch == "vit_small" and args.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif args.arch == "vit_small" and args.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif args.arch == "vit_base" and args.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif args.arch == "vit_base" and args.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            dinov2_model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    run_pipeline(
        encoder_model=encoder_model,
        dinov2_model=dinov2_model,
        output_dir=args.output_dir,
        train_dataset_str=args.train_dataset_str,
        val_dataset_str=args.val_dataset_str,
        test_dataset_strs=args.test_dataset_strs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        epoch_length=args.epoch_length,
        num_workers=args.num_workers,
        save_checkpoint_frequency=args.save_checkpoint_frequency,
        eval_period_iterations=args.eval_period_iterations,
        learning_rates=args.learning_rates,
        autocast_dtype=autocast_dtype,
        resume=not args.no_resume,
        classifier_fpath=args.classifier_fpath,
        val_metric_type=args.val_metric_type,
        test_metric_types=args.test_metric_types,
        val_class_mapping_fpath=args.val_class_mapping_fpath,
        test_class_mapping_fpaths=args.test_class_mapping_fpaths,
    )

    sys.exit(0)
