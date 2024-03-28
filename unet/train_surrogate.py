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


sys.path.append("/home/students/tnguyen/masterthesis")

from unet.unet_model import UNet
from unet.evaluate import evaluate


class SurrogateDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, target_dir, image_size):
        super(SurrogateDataset, self).__init__()
        self.img_list = img_list
        self.target_dir = target_dir
        self.img_transform = tv_transforms.Compose([
            tv_transforms.Resize((image_size, image_size)),
            tv_transforms.Grayscale(num_output_channels=3),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        
        image_dir, image_name = os.path.split(img_path)
        image_dir = image_dir.split('/')
        image_dir = image_dir[-2:]
        target_path = self.target_dir + image_dir[1] + '/' + image_name
        with open(target_path, 'rb') as f:
            target = Image.open(f)
            target = target.convert('RGB')

        return self.img_transform(img), self.img_transform(target)
    
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 0,
        # weight_decay: float = 1e-8,
        # momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        image_size: int = 224,
        patch_size: int = 14,
):
    # 1. Create dataset
    train_dir_path = r'/images/PublicDatasets/imagenet/train/'
    train_files = glob.glob(os.path.join(train_dir_path, "*/*.jpg"))
    if not args.run_on_cluster:
        shuffle(train_files)
        train_files = train_files[:5000]
    train_target_dir_path = r'/images/innoretvision/eye/imagenet_patch/p2p/train/'
    
    val_dir_path = r'/images/PublicDatasets/imagenet_shared/val/'
    val_files = glob.glob(os.path.join(val_dir_path, "*/*.JPEG"))
    if not args.run_on_cluster:
        shuffle(val_files)
        val_files = val_files[:5000]
    val_target_dir_path = r'/images/innoretvision/eye/imagenet_patch/p2p/val/'

    train_set = SurrogateDataset(train_files, train_target_dir_path, image_size)
    val_set = SurrogateDataset(val_files, val_target_dir_path, image_size)
    n_train = len(train_set)
    n_val = len(val_set)
    
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.MSELoss()
    global_step = 0

    # 5. Begin training
    n_patches = (image_size // patch_size) ** 2
    """Load checkpoint if neccesary"""
    epoch_start = 1
    ckpt_path = args.output_dir + '/training_checkpoint.pth'
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_start = int(checkpoint['epoch']) + 1
    for epoch in range(epoch_start, epochs + 1):
        print(f"lr {optimizer.param_groups[0]['lr']}")
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train*n_patches, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch[0], batch[1]

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                # save_image(images, args.output_dir + 'sample_image.jpg')
                # save_image(true_masks, args.output_dir + 'sample_target.jpg')
                # print(f'images shape {images.shape}')
                images = pypatchify.patchify_to_batches(images, (3, patch_size, patch_size), batch_dim=0)
                # save_image(images[100], args.output_dir + 'sample_image_p100.jpg')
                # print(f'images shape {images.shape}')
                true_masks = pypatchify.patchify_to_batches(true_masks, (3, patch_size, patch_size), batch_dim=0)
                # print(f'true_masks shape {true_masks.shape}')
                # save_image(true_masks[100], args.output_dir + 'sample_target_p100.jpg')

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # save_image(masks_pred[100], args.output_dir + 'sample_predict_p100.jpg')
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        # loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item() / (n_patches),
                    'step': global_step,
                    'epoch': epoch
                })
                if not args.run_on_cluster:
                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{'loss (batch)': loss.item() / (n_patches)})
                else:
                    if global_step % 5000 == 0:
                        print(f'loss (batch) {loss.item() / (n_patches)}')

                # Evaluation round
                if (global_step > 5) and (global_step % (n_train // batch_size) == 0):
                    histograms = {}
                    for tag, value in model.named_parameters():
                        tag = tag.replace('/', '.')
                        if not (torch.isinf(value) | torch.isnan(value)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(model, patch_size, n_patches, criterion, val_loader, device, amp)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    try:
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })
                    except:
                        pass

        """General checkpoint"""
        train_loss = epoch_loss / (len(train_loader) * n_patches)
        metrics_file_path = os.path.join(args.output_dir, "results.json")
        with open(metrics_file_path, "a") as f:
            f.write(f"epoch: {epoch}\n")
            f.write(json.dumps({'lr': optimizer.param_groups[0]['lr']}) + "\n")
            f.write(json.dumps({'train_loss': train_loss}) + "\n")
            f.write(json.dumps({'val_loss': val_score.item()}) + "\n")
            f.write("\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_score.item(),
            }, str(args.output_dir + '/training_checkpoint.pth'))

        if save_checkpoint:
            state_dict = model.state_dict()
            torch.save(state_dict, str(args.output_dir + '/checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target')
    parser.add_argument('--image-size', '-i', metavar='I', type=int, default=224, help='Full image size')
    parser.add_argument('--patch-size', '-p', metavar='P', type=int, default=14, help='Patch size')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-w', metavar='WD', type=float, default=0, help='Weight decay', dest='wd')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--output-dir', '-o', type=str, default='/work/scratch/tnguyen/unet/0/', help='Dir path to save checkpoint')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument("--run_on_cluster", action="store_true", help="Run on cluster or local machine. Default: local machine.")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(device=device)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            image_size=args.image_size,
            patch_size=args.patch_size,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.wd,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            image_size=args.image_size,
            patch_size=args.patch_size,
        )
