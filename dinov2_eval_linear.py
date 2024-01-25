import logging
import os
import sys

from functools import partial
from typing import Any, Callable, List, Optional, TypeVar

import torch
import torch.nn as nn
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import webdataset as wds
import wids

sys.path.append("/home/students/tnguyen/masterthesis/dinov2_lib")

from dinov2.logging import setup_logging
import dinov2.distributed as distributed
from dinov2.eval.linear import get_args_parser as get_linear_args_parser
from dinov2.eval.linear import has_ddp_wrapper, remove_ddp_wrapper, setup_linear_classifiers, evaluate_linear_classifiers, _pad_and_collate, LinearClassifier, AllClassifiers, LinearPostprocessor, scale_lr, test_on_datasets
from dinov2.eval.setup import setup_and_build_model
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
from dinov2.data.datasets import ImageNet, ImageNet22k
from dinov2.data.samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


def wds_gen(url, split = 'train'):
    pil_dataset = (
        wds.WebDataset(
        # wids.ShardListDataset(
            url,
        )
        .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg", "cls")
    )
    transform = make_transform(split)

    def preprocess(sample):
        image, label = sample
        # image, label = sample[".jpg"], sample[".cls"]
        return transform(image), label

    wds_dataset = pil_dataset.map(preprocess)
    return wds_dataset


def make_transform(split):
    if split == 'train':
        return make_classification_train_transform()
    else:
        return make_classification_eval_transform()


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
    if test_dataset_str == 'ImageNetWds':
        test_transform = make_classification_eval_transform()
        pil_dataset = (
            ImageNetWds(
            # wids.ShardListDataset(
                '/images/innoretvision/eye/imagenet_patch/val/imagenet-val-{000000..000006}.tar',
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
            transform=make_classification_eval_transform(),
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

        features = feature_model(data)
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
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            print("lr", optimizer.param_groups[0]["lr"])

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

    train_transform = make_classification_train_transform()
    if train_dataset_str == 'ImageNetWds':
        pil_dataset = (
            ImageNetWds(
            # wids.ShardListDataset(
                '/images/innoretvision/eye/imagenet_patch/train/imagenet-train-{000000..000146}.tar',
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
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
    if train_dataset_str == 'ImageNetWds':
        for image, label in train_dataset:
            break
        sample_output = feature_model(image.unsqueeze(0).cuda())
    else:
        sample_output = feature_model(train_dataset[0][0].unsqueeze(0).cuda())

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        sample_output,
        n_last_blocks_list,
        learning_rates,
        batch_size,
        training_num_classes,
    )

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
    results_dict = {}
    if len(test_dataset_strs) > 1 or test_dataset_strs[0] != val_dataset_str:
        results_dict = test_on_datasets(
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


if __name__ == "__main__":
    description = "DINOv2 linear evaluation"
    args_parser = get_linear_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))
