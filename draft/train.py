from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import argparse
import gc
from model_ViT import *
from dataset_imagenet import ImageNetData
from transformers import get_cosine_schedule_with_warmup
from utils.logging_utils import Log

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes):
    since = time.time()

    best_model_wts = model.state_dict()

    for epoch in range(args.start_epoch+1,num_epochs):

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            tic_batch = time.time()
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # wrap them in Variable
                if use_gpu:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # forward
                if phase == 'train':
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    outputs = model(inputs).logits
                else:
                    with torch.no_grad():
                        outputs = model(inputs).logits
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # statistics
                running_loss += float(loss.item())
                running_corrects += torch.sum(preds == labels.data)

                batch_loss = running_loss / ((i+1)*args.batch_size)
                batch_acc = running_corrects / ((i+1)*args.batch_size)

                if phase == 'train' and i%args.print_freq == 0:
                    log.logger.info('[Epoch {}/{}]-[batch:{}/{}] lr:{:.4f} {} Loss: {:.6f}  Acc: {:.4f}  Time: {:.4f}batch/sec'.format(
                          epoch, num_epochs - 1, i, round(dataset_sizes[phase]/args.batch_size)-1, scheduler.get_last_lr()[0], phase, batch_loss, batch_acc, \
                        args.print_freq/(time.time()-tic_batch)))
                    tic_batch = time.time()

                # clear the cache
                del inputs, labels
                gc.collect()
                torch.cuda.empty_cache()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            log.logger.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

        if (epoch+1) % args.save_epoch_freq == 0:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model, os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth.tar"))

    time_elapsed = time.time() - since
    log.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch implementation of fixationViT")
    parser.add_argument('--data-dir', type=str, default=r'/images/PublicDatasets/imagenet')
    parser.add_argument('--data-limit', type=int, default=50000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    args = parser.parse_args()

    '''Set up log'''
    output_dir = '/home/students/tnguyen/results/fixation_ViT/'
    log = Log(output_dir)

    # read data
    dataloaders, dataset_sizes = ImageNetData(args.data_dir, args.data_limit, False, args)

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    use_gpu = False
    log.logger.info("use_gpu:{}".format(use_gpu))

    # get model
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 10000
    model = fixation_ViT.from_pretrained(
        "/home/students/tnguyen/masterthesis/huggingface/vit-base-patch16-224",
        num_labels = args.num_class,
        patch_embeddings_type = 'conv_cat',  # '', conv_cat, linear, or conv_linear
        lr = args.lr, 
        weight_decay = WEIGHT_DECAY, 
        warmup = WARMUP_STEPS,
    )
    # freezing everything
    for param in model.parameters():
        param.requires_grad = False
    # only train pooler and classifier layers
    model.pooler.requires_grad_(True)
    model.classifier.requires_grad_(True)

    if args.resume:
        if os.path.isfile(args.resume):
            log.logger.info(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
            model.load_state_dict(base_dict)
        else:
            log.logger.info(("=> no checkpoint found at '{}'".format(args.resume)))

    if use_gpu:
        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = torch.optim.Adam(
        model.parameters(),
        lr=model.lr,
        weight_decay=model.weight_decay,
    )

    # Cosine warmup scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer_ft,
        num_warmup_steps=model.warmup,
        num_training_steps=len(dataloaders),
    )

    model = train_model(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer_ft,
        scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes
    )
