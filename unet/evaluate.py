import torch
import torch.nn.functional as F
import pypatchify
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, patch_size, n_patches, criterion, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    val_loss = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            image = pypatchify.patchify_to_batches(image, (3, patch_size, patch_size), batch_dim=0)
            mask_true = pypatchify.patchify_to_batches(mask_true, (3, patch_size, patch_size), batch_dim=0)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                val_loss += criterion(mask_pred.squeeze(1), mask_true.float())
            else:
                val_loss += criterion(mask_pred, mask_true)

    net.train()
    return val_loss / max(num_val_batches * n_patches, 1)
