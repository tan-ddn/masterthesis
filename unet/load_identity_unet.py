import sys
import torch
import torchvision
import logging

from functools import partial
from torch import nn
from torchvision import transforms as pth_transforms

sys.path.append("/home/students/tnguyen/masterthesis")
sys.path.append("/home/students/tnguyen/masterthesis/dinov2_lib")

from dinov2_lib.dinov2.eval.metrics import MetricType, build_metric
from dinov2_lib.dinov2.eval.linear import _pad_and_collate
from dinov2_lib.dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2_tan.data_transforms import make_classification_eval_transform, make_classification_train_transform
from unet.train_encoder import ImageNetWds
from unet.unet_model import UNet, UNet2


PATH = r"/work/scratch/tnguyen/unet/identity_encoder/3/checkpoint_epoch10.pth"

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

    ckpt = torch.load(PATH)
    # for key, value in ckpt.items():
        # if key not in ["optimizer", "scheduler", "iteration"]:
        #     print(f"key {key}")
    # model = ckpt["model"]
    # for key, value in model.items():
    #     print(key)
    # sys.exit(0)

    val_data_loader, val_dataset = make_eval_data_loader("ImageNetWds", 1, MetricType.MEAN_ACCURACY)

    model = UNet2(n_channels=1, n_classes=1, bilinear=False).to(device)
    # model = UNet(n_channels=1, n_classes=1, bilinear=False).to(device)
    print(f'Encoder:\n'
                f'\t{model.n_channels} input channels\n'
                f'\t{model.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    
    trained_weights = torch.tensor([], device=device)
    for name, p in model.named_parameters():
        print(f"{name}: {p.requires_grad}, {p.grad}")
        p_1d = p.view(p.nelement())
        trained_weights = torch.cat((trained_weights, p_1d))
    print(f"trained weights shape {trained_weights.shape}")
    print(f"max min mean std: {torch.max(trained_weights), torch.min(trained_weights), torch.mean(trained_weights), torch.std(trained_weights)}")


    # model.load_state_dict(ckpt)
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad = False
        
    # to_grayscale = pth_transforms.Grayscale(num_output_channels=1)
    
    # transform = pth_transforms.Compose([
    #     pth_transforms.Resize((14, 14)),
    #     # pth_transforms.ToTensor(),
    #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])

    # for i, (image, label) in enumerate(val_dataset):
    #     if i > 5:
    #         break
    #     else:
    #         print(f"Image shape, max, and min {image.shape, image.max(), image.min()}")
    
    #         image = to_grayscale(image.to(device))
    #         # image = transform(image.to(device))
    #         torchvision.utils.save_image(image, r"/home/students/tnguyen/masterthesis/unet/identity/"+str(i)+r"original_image.jpg")
    #         # torchvision.utils.save_image(image, r"/home/students/tnguyen/masterthesis/unet/identity/resized_"+str(i)+r"original_image.jpg")
    #         image = image.unsqueeze(0).cuda()
    #         output = model(image)
    #         # output = encoder_model(image)
    #         torchvision.utils.save_image(output[0], r"/home/students/tnguyen/masterthesis/unet/identity/"+str(i)+r"after_i_unet_image.jpg")
    #         # torchvision.utils.save_image(output[0], r"/home/students/tnguyen/masterthesis/unet/identity/resized_"+str(i)+r"after_i_unet_image.jpg")
