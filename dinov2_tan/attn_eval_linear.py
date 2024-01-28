# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import logging
import os, psutil
import sys
import argparse
import requests
import json
from io import BytesIO

from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as pth_transforms
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import webdataset as wds
import wids
import gc


sys.path.append("/home/students/tnguyen/masterthesis")

import dino.utils
import vision_transformer as vits
from fixation_classifiers import Fixation
from dinov2_lib.dinov2.eval.setup import get_autocast_dtype
import dinov2_lib.dinov2.utils.utils as dinov2_utils
from dinov2_lib.dinov2.utils.config import setup
import dinov2_lib.dinov2.distributed as distributed
from dinov2_lib.dinov2.eval.linear import has_ddp_wrapper, remove_ddp_wrapper, LinearPostprocessor, _pad_and_collate, AllClassifiers, scale_lr, setup_linear_classifiers
from dinov2_lib.dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2_lib.dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
from dinov2_lib.dinov2.eval.utils import ModelWithIntermediateLayers
from dinov2_lib.dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2_lib.dinov2.eval.metrics import MetricType, build_metric
from dinov2_lib.dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
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
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
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
        type=int,
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
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument("--image_size", default=518, type=int, help="Resize image.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--fixation_grayscale", action="store_true", help="Fixations in grayscale or rgb. Default is rgb.")
    parser.add_argument("--fixation_top", default=0.1, type=float, help="Percentage of fixations with top attention score.")
    parser.add_argument("--run_on_cluster", action="store_true", help="Run on cluster or local machine. Default: local machine.")
    parser.set_defaults(
        train_dataset_str="ImageNetWds",
        val_dataset_str="ImageNetWds",
        test_dataset_strs=None,
        epochs=10,
        batch_size=512,
        num_workers=1,
        epoch_length=2501,
        save_checkpoint_frequency=1,
        eval_period_iterations=2501,
        learning_rates=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1],
        val_metric_type=MetricType.MEAN_ACCURACY,
        test_metric_types=None,
        classifier_fpath=None,
        val_class_mapping_fpath=None,
        test_class_mapping_fpaths=[None],
        patch_size=14,
        image_size=518,
        checkpoint_key="teacher",
        fixation_grayscale=False,
        fixation_top=0.1,
        run_on_cluster=False,
    )
    return parser


class ModelLastSelfAttentionFixation(nn.Module):
    def __init__(self, feature_model, args, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.fixation_layer = Fixation(
            device=args.device,
            img_size=args.image_size,
            patch_size=args.patch_size,
            fixation_grayscale=args.fixation_grayscale,
            top=args.fixation_top,
        )
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.no_grad():
            with self.autocast_ctx():
                outputs, attentions = self.feature_model.get_last_selfattention(
                    images, self.n_last_blocks
                )
                fixations = self.fixation_layer(attentions, images)
        return outputs, fixations


# class LinearClassifier(nn.Module):
#     """Linear layer to train on top of frozen features"""

#     def __init__(self, out_dim, num_classes=1000):
#         super().__init__()
#         self.out_dim = out_dim
#         self.num_classes = num_classes
#         self.linear = nn.Linear(out_dim, num_classes)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()

#     def forward(self, x_tokens_list):
#         return self.linear(x_tokens_list)


# def setup_linear_classifiers(sample_output, learning_rates, batch_size, num_classes=1000):
#     linear_classifiers_dict = nn.ModuleDict()
#     optim_param_groups = []
#     for _lr in learning_rates:
#         lr = scale_lr(_lr, batch_size)
#         out_dim = sample_output.shape[1]
#         linear_classifier = LinearClassifier(
#             out_dim, num_classes=num_classes
#         )
#         linear_classifier = linear_classifier.cuda()
#         linear_classifiers_dict[
#             f"classifier_lr_{lr:.5f}".replace(".", "_")
#         ] = linear_classifier
#         optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

#     linear_classifiers = AllClassifiers(linear_classifiers_dict)
#     if distributed.is_enabled():
#         linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

#     return linear_classifiers, optim_param_groups


@torch.no_grad()
def evaluate(
    fixation_model,
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    fixation_model.eval()
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        # outputs = model(samples.to(device))

        # outputs, fixations = model(samples.to(device))
        _, fixations = fixation_model(samples.to(device))
        outputs = model(fixations)

        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            # metric_inputs = postprocessors[k](fixations, targets)
            metric.update(**metric_inputs)

    # clear memory
    del fixations, outputs

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


@torch.no_grad()
def evaluate_linear_classifiers(
    fixation_model,
    feature_model,
    linear_classifiers,
    data_loader,
    metric_type,
    metrics_file_path,
    training_num_classes,
    iteration,
    prefixstring="",
    class_mapping=None,
    best_classifier_on_val=None,
):
    logger.info("running validation !")

    num_classes = len(class_mapping) if class_mapping is not None else training_num_classes
    metric = build_metric(metric_type, num_classes=num_classes)
    postprocessors = {k: LinearPostprocessor(v, class_mapping) for k, v in linear_classifiers.classifiers_dict.items()}
    metrics = {k: metric.clone() for k in linear_classifiers.classifiers_dict}

    _, results_dict_temp = evaluate(
        fixation_model,
        feature_model,
        data_loader,
        postprocessors,
        metrics,
        torch.cuda.current_device(),
    )

    logger.info("")
    results_dict = {}
    max_accuracy = 0
    best_classifier = ""
    for i, (classifier_string, metric) in enumerate(results_dict_temp.items()):
        logger.info(f"{prefixstring} -- Classifier: {classifier_string} * {metric}")
        if (
            best_classifier_on_val is None and metric["top-1"].item() > max_accuracy
        ) or classifier_string == best_classifier_on_val:
            max_accuracy = metric["top-1"].item()
            best_classifier = classifier_string

    results_dict["best_classifier"] = {"name": best_classifier, "accuracy": max_accuracy}

    logger.info(f"best classifier: {results_dict['best_classifier']}")

    if distributed.is_main_process():
        with open(metrics_file_path, "a") as f:
            f.write(f"iter: {iteration}\n")
            for k, v in results_dict.items():
                f.write(json.dumps({k: v}) + "\n")
            f.write("\n")

    return results_dict


def test_on_datasets(
    fixation_model,
    feature_model,
    linear_classifiers,
    test_dataset_strs,
    batch_size,
    num_workers,
    test_metric_types,
    metrics_file_path,
    training_num_classes,
    iteration,
    best_classifier_on_val,
    prefixstring="",
    test_class_mappings=[None],
):
    results_dict = {}
    for test_dataset_str, class_mapping, metric_type in zip(test_dataset_strs, test_class_mappings, test_metric_types):
        logger.info(f"Testing on {test_dataset_str}")
        test_data_loader = make_eval_data_loader(test_dataset_str, batch_size, num_workers, metric_type)
        dataset_results_dict = evaluate_linear_classifiers(
            fixation_model,
            feature_model,
            remove_ddp_wrapper(linear_classifiers),
            test_data_loader,
            metric_type,
            metrics_file_path,
            training_num_classes,
            iteration,
            prefixstring="",
            class_mapping=class_mapping,
            best_classifier_on_val=best_classifier_on_val,
        )
        results_dict[f"{test_dataset_str}_accuracy"] = 100.0 * dataset_results_dict["best_classifier"]["accuracy"]
    return results_dict


class ImageNetWds(wds.WebDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.split = 'train'

    def __len__(self):
        if self.split == 'train':
            return 1281167
        else:
            return 50000

    def set_split(self, split='train'):
        self.split = split
        return self
        

def make_eval_data_loader(test_dataset_str, batch_size, num_workers, metric_type):
    resize_size = int(args.image_size * 1.15)
    if test_dataset_str == 'ImageNetWds':
        test_transform = make_classification_eval_transform(resize_size=resize_size, crop_size=args.image_size)
        test_data_path = r'/images/innoretvision/eye/imagenet_patch/val/imagenet-val-{000000..000003}.tar'
        if args.run_on_cluster:
            test_data_path = r'/images/innoretvision/eye/imagenet_patch/val/imagenet-val-{000000..000006}.tar'
        pil_dataset = (
            ImageNetWds(
            # wids.ShardListDataset(
                test_data_path,
            )
            .set_split('val')
            .shuffle(5000)
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
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_type=None,
            drop_last=False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
        )
    else:
        test_dataset = make_dataset(
            dataset_str=test_dataset_str,
            transform=make_classification_eval_transform(resize_size=resize_size, crop_size=args.image_size),
        )
        test_data_loader = make_data_loader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler_type=SamplerType.DISTRIBUTED,
            drop_last=False,
            shuffle=False,
            persistent_workers=False,
            collate_fn=_pad_and_collate if metric_type == MetricType.IMAGENET_REAL_ACCURACY else None,
        )
    return test_data_loader


def eval_linear(
    *,
    fixation_model,
    feature_model,
    linear_classifiers,
    train_data_loader,
    val_data_loader,
    metrics_file_path,
    optimizer,
    scheduler,
    output_dir,
    max_iter,
    checkpoint_period,  # In number of iter, creates a new file every period
    running_checkpoint_period,  # Period to update main checkpoint file
    eval_period,
    metric_type,
    training_num_classes,
    resume=True,
    classifier_fpath=None,
    val_class_mapping=None,
):
    checkpointer = Checkpointer(linear_classifiers, output_dir, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(classifier_fpath or "", resume=resume).get("iteration", -1) + 1

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    for data, labels in metric_logger.log_every(
        train_data_loader,
        500,
        header,
        max_iter,
        start_iter,
    ):
        data = data.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        # features = feature_model(data)
        # outputs = linear_classifiers(features)
        # features, fixations = feature_model(data)
        # outputs = linear_classifiers(fixations)
        _, fixations = fixation_model(data)
        features = feature_model(fixations)
        outputs = linear_classifiers(features)

        losses = {f"loss_{k}": nn.CrossEntropyLoss()(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()
        scheduler.step()

        # log
        if iteration % 100 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            print("lr", optimizer.param_groups[0]["lr"])
            # clear memory
            del fixations, features, outputs
            ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            print(f"ram usage: {ram_usage}")

        if iteration - start_iter > 5:
            if iteration % running_checkpoint_period == 0:
                torch.cuda.synchronize()
                if distributed.is_main_process():
                    logger.info("Checkpointing running_checkpoint")
                    periodic_checkpointer.save("running_checkpoint_linear_eval", iteration=iteration)
                torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        if eval_period > 0 and (iteration + 1) % eval_period == 0 and iteration != max_iter - 1:
            _ = evaluate_linear_classifiers(
                fixation_model=fixation_model,
                feature_model=feature_model,
                linear_classifiers=remove_ddp_wrapper(linear_classifiers),
                data_loader=val_data_loader,
                metrics_file_path=metrics_file_path,
                prefixstring=f"ITER: {iteration}",
                metric_type=metric_type,
                training_num_classes=training_num_classes,
                iteration=iteration,
                class_mapping=val_class_mapping,
            )
            torch.cuda.synchronize()

        iteration = iteration + 1

    val_results_dict = evaluate_linear_classifiers(
        fixation_model=fixation_model,
        feature_model=feature_model,
        linear_classifiers=remove_ddp_wrapper(linear_classifiers),
        data_loader=val_data_loader,
        metrics_file_path=metrics_file_path,
        metric_type=metric_type,
        training_num_classes=training_num_classes,
        iteration=iteration,
        class_mapping=val_class_mapping,
    )
    return val_results_dict, feature_model, linear_classifiers, iteration


def run_eval_linear(
    model,
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

    train_transform = make_classification_train_transform(crop_size=args.image_size)
    if train_dataset_str == 'ImageNetWds':
        train_data_path = r'/images/innoretvision/eye/imagenet_patch/train/imagenet-train-{000000..000070}.tar'
        if args.run_on_cluster:
            train_data_path = r'/images/innoretvision/eye/imagenet_patch/train/imagenet-train-{000000..000146}.tar'
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

    n_last_blocks_list = [1, 4]
    n_last_blocks = max(n_last_blocks_list)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    fixation_model = ModelLastSelfAttentionFixation(model, args, 2, autocast_ctx).to(args.device)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    if train_dataset_str == 'ImageNetWds':
        for image, label in train_dataset:
            break
        # sample_output, fixations = feature_model(image.unsqueeze(0).cuda())
        _, fixations = fixation_model(image.unsqueeze(0).cuda())
        sample_output = feature_model(fixations)
    else:
        # sample_output, fixations = feature_model(train_dataset[0][0].unsqueeze(0).cuda())
        _, fixations = fixation_model(train_dataset[0][0].unsqueeze(0).cuda())
        sample_output = feature_model(fixations)

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
    val_data_loader = make_eval_data_loader(val_dataset_str, batch_size, num_workers, val_metric_type)

    checkpoint_period = save_checkpoint_frequency * epoch_length

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
            fixation_model=fixation_model,
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
            fixation_model,
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


def main(args):
    model, autocast_dtype = setup_and_build_model(args)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
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
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    run_eval_linear(
        model=model,
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
    return 0


def build_model(args, only_teacher=False, img_size=224):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)


def build_model_for_eval(config, pretrained_weights):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup(args)
    model = build_model_for_eval(config, args.pretrained_weights)
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype


def find_coordinates_from_idx(columns, id):
    y = id // columns
    x = id % columns
    return (x, y)


def visualize_fixations(args):
    # build model
    # model = vits.__dict__[args.arch](img_size=args.image_size[0], patch_size=args.patch_size)

    model, autocast_dtype = setup_and_build_model(args)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(args.device)
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
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
            model.load_state_dict(state_dict, strict=True)
        else:
            print("There is no reference weights available for this model => We use random weights.")

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.image_size, args.image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    print(f"img shape {img.shape}")

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size
    
    """For debuging"""
    img = torch.tile(img, (2, 1, 1, 1)).to(args.device)
    print(f"Batch img shape {img.shape}")
    # outputs, attentions = model.get_last_selfattention(img)
    # print(f"attentions shape {attentions.shape}")
    # fixation_layer = Fixation(
    #     device=device,
    #     images=img,
    #     img_size=args.image_size,
    #     patch_size=args.patch_size,
    #     fixation_grayscale=args.fixation_grayscale,
    #     top=args.fixation_top
    # )
    # fixations = fixation_layer(attentions)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
    feature_model = ModelLastSelfAttentionFixation(model, args, 4, autocast_ctx)
    sample_output, fixations = feature_model(img)
    for i in range(len(sample_output)):
        print(f"last layer {i}")
        out, cls = sample_output[i]
        print(f"out shape {out[0].shape}")
        print(f"cls shape {cls[0].shape}")
    # fixations = fixations.reshape((img.shape[0], 1, img.shape[2], img.shape[3]))
    print(f"fixations shape {fixations.shape}")
    """Visualize the fixations"""
    image_dir, image_name = os.path.split(args.image_path)  # image_dir ~ /images/PublicDatasets/imagenet/train/n02823428
    image_dir = image_dir.split('/')
    image_dir = image_dir[-2:]
    image_name = image_name.split('.')
    saved_dir = args.output_dir + image_dir[0] + '_50/' + image_dir[1] + '/'
    os.makedirs(saved_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(fixations, normalize=True, scale_each=True), os.path.join(saved_dir, image_name[0] + '_b_f_grsc.' + image_name[1]))


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    description = "DINOv2 Visualize Self-Attention maps"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    args.device = device

    sys.exit(main(args))

    parser = argparse.ArgumentParser('Visualize Self-Attention Linear Classification')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=518, type=int, help="Resize image.")
    parser.add_argument("--fixation_grayscale", action="store_true", help="Fixations in grayscale or rgb. Default is rgb")
    parser.add_argument("--fixation_top", default=0.1, type=float, help="Percentage of fixations with top attention score")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--run_on_cluster", action="store_true", help="Run on cluster or local machine. Default: local machine.")
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    args = parser.parse_args()
    args.device = device
    sys.exit(visualize_fixations(args))
