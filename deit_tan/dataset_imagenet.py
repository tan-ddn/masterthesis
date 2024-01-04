import os
import torch
import torchvision
import torchvision.io as tv_io
import scipy.io as scio
import cv2
import pickle
from PIL import Image
from torchvision import transforms, datasets
from tqdm.auto import tqdm

from torchvision import transforms as pth_transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def ImageNetData(data_dir : str = '', data_limit : int = -1, return_dataset : bool = True, args = None):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    #image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])

    image_datasets['train'] = ImageNetTrainDataSet(
        os.path.join(data_dir, 'train'),
        os.path.join(data_dir, 'ILSVRC2012_devkit_t12', 'data', 'meta.mat'),
        data_transforms['train'],
        data_limit
    )
    image_datasets['val'] = ImageNetValDataSet(
        os.path.join(data_dir, 'val'),
        os.path.join(data_dir, 'ILSVRC2012_devkit_t12', 'data','ILSVRC2012_validation_ground_truth.txt'),
        data_transforms['val'],
        data_limit
    )

    if return_dataset:
        return image_datasets['train'], image_datasets['val']
    else:
        # wrap your data and label into Tensor
        dataloaders = {
            x: torch.utils.data.DataLoader(
                image_datasets[x],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers
            ) for x in ['train', 'val']
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        return dataloaders, dataset_sizes

class ImageNetTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_label, data_transforms, data_limit):
        label_array = scio.loadmat(img_label)['synsets']
        label_dic = {}
        for i in  range(1000):
            label_dic[label_array[i][0][1][0]] = i
        self.img_path = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.label_dic = label_dic
        self.root_dir = root_dir
        self.imgs = self._make_dataset(data_limit)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data, label = self.imgs[item]
        img = Image.open(data).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label

    def _make_dataset(self, data_limit):
        class_to_idx = self.label_dic
        images = []
        dir = os.path.expanduser(self.root_dir)
        progress_bar = tqdm(range(1000))
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            progress_bar.update(1)
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self._is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

        return images[:data_limit]

    def _is_image_file(self, filename):
        """Checks if a file is an image.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

class ImageNetValDataSet(torch.utils.data.Dataset):
    def __init__(self, img_path, img_label, data_transforms, data_limit):
        self.data_transforms = data_transforms
        img_names = os.listdir(img_path)
        img_names.sort()
        self.img_path = [os.path.join(img_path, img_name) for img_name in img_names]
        with open(img_label,"r") as input_file:
            lines = input_file.readlines()
            self.img_label = [(int(line)-1) for line in lines]

        self.img_path = self.img_path[:data_limit]
        self.img_label = self.img_label[:data_limit]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert('RGB')
        label = self.img_label[item]
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label
    

def imagenetfix_data(args, defaultValues):
    train_dataset = ImagenetFixationDataset(
        # root=defaultValues['train_root'],
        root=defaultValues['image_dir']+'train/',
        extensions=args.extensions,
        patch_dir=defaultValues['image_dir']+'train/',
        max_fix_length=defaultValues['max_fix_length'],
        channels=args.num_channel,
        patch_size=defaultValues['patch_size'],
        data_limit=args.data_limit
    )
    val_dataset = ImagenetFixationDataset(
        # root=defaultValues['val_root'],
        root=defaultValues['image_dir']+'val/',
        extensions=args.extensions,
        patch_dir=defaultValues['image_dir']+'val/',
        max_fix_length=defaultValues['max_fix_length'],
        channels=args.num_channel,
        patch_size=defaultValues['patch_size'],
        data_limit=args.data_limit
    )  
    return train_dataset, val_dataset
    # trainloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=True,
    #     pin_memory=False,
    #     # num_workers=8,
    # )
    # valloader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=args.val_batch_size,
    #     shuffle=True,
    #     pin_memory=False,
    #     # num_workers=8,
    # )
    # dataset_sizes = {
    #     'train': len(train_dataset),
    #     'val': len(val_dataset),
    # }
    # return trainloader, valloader, dataset_sizes

def path_loader(path: str) -> str:
    return path

class ImagenetFixationDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, root : None,
                 patch_dir : str,
                 max_fix_length : int,
                 channels : int,
                 patch_size : tuple,
                 extensions : tuple = (".pt"),
                 data_limit: int | None = None,
                 device : str = 'cpu', ):
        super().__init__(root, loader=path_loader, extensions=extensions)
        self.root = root
        self.patch_dir = patch_dir
        self.max_fix_length = int(max_fix_length)
        self.channels = channels
        self.patch_size = patch_size
        self.device = device
        if data_limit is not None:
            self.samples = self.samples[:data_limit]
            self.targets = self.targets[:data_limit]

    def __getitem__(self, index):
        """Generates one sample of data"""
        """Load data and get label"""
        path, target = self.samples[index]
        # image_dir, image_name = os.path.split(path)  # image_dir ~ /images/PublicDatasets/imagenet/train/n02823428
        # image_dir = image_dir.split('/')
        # image_name = image_name.split('.')
        # fixations = torch.zeros(
        #     (self.max_fix_length, self.channels, self.patch_size[0], self.patch_size[1]),
        #     device=self.device,
        # )
        # transform = transforms.ToTensor()
        # # fixations = []
        # for i in range(self.max_fix_length):
        #     patch_name = image_name[0] + '_' + str(i) + '_16x16.' + image_name[1]
        #     # patch = tv_io.read_image(
        #     #     self.patch_dir + image_dir[-1] + '/' + patch_name,
        #     #     mode=tv_io.ImageReadMode.GRAY
        #     # )  # patch shape = (1, 16, 16)
        #     patch = cv2.imread(self.patch_dir + image_dir[-1] + '/' + patch_name)
        #     patch = transform(patch)
        #     print(f'patch shape {patch.shape}')
        #     fixations[i] = patch
        #     # fixations.append(patch)
        # # fixations = torch.stack(fixations, dim=0).to(self.device)
        saved_fixations = torch.load(path)
        return saved_fixations, target
        # fixations = saved_fixations[:self.max_fix_length]
        return fixations, target
    

if __name__ == '__main__':
    """Create dataset with fixations"""
    defaultValues = {
        'train_root': r'/images/PublicDatasets/imagenet/train/',
        'val_root': r'/images/PublicDatasets/imagenet_shared/val/',
        'image_dir': r'/work/scratch/tnguyen/images/imagenet/patches/',
        'max_fix_length': 10,
        'patch_size': (16, 16),
    }
    # train_dataset = ImagenetFixationDataset(
    #     # root=defaultValues['train_root'],
    #     root=defaultValues['image_dir']+'train/',
    #     extensions=(".pt"),
    #     patch_dir=defaultValues['image_dir']+'train/',
    #     max_fix_length=defaultValues['max_fix_length'],
    #     channels=1,
    #     patch_size=defaultValues['patch_size'],
    # )
    # val_dataset = ImagenetFixationDataset(
    #     # root=defaultValues['val_root'],
    #     root=defaultValues['image_dir']+'val/',
    #     extensions=(".pt"),
    #     patch_dir=defaultValues['image_dir']+'val/',
    #     max_fix_length=defaultValues['max_fix_length'],
    #     channels=1,
    #     patch_size=defaultValues['patch_size'],
    # )  
    dataset_train_file = defaultValues['image_dir'] + 'dataset_train.pickle'
    # with open(dataset_train_file, 'wb') as handle:
    #     pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    dataset_val_file = defaultValues['image_dir'] + 'dataset_val.pickle'
    # with open(dataset_val_file, 'wb') as handle:
    #     pickle.dump(val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    """Create dataset with image size: 160x160x3"""
    transform = pth_transforms.Compose([
        pth_transforms.Resize((160, 160)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    train_root = r'/images/PublicDatasets/imagenet/train/'
    train_dataset = torchvision.datasets.ImageFolder(train_root, transform=transform)
    dataset_train_file = defaultValues['image_dir'] + 'dataset_train_160.pickle'
    with open(dataset_train_file, 'wb') as handle:
        pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    val_root = r'/images/PublicDatasets/imagenet_shared/val/'
    val_dataset = torchvision.datasets.ImageFolder(val_root, transform=transform)
    dataset_val_file = defaultValues['image_dir'] + 'dataset_val_160.pickle'
    with open(dataset_val_file, 'wb') as handle:
        pickle.dump(val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # """Create dataset with image size: 160x160x1"""
    # transform = pth_transforms.Compose([
    #     pth_transforms.Resize((160, 160)),
    #     pth_transforms.Grayscale(num_output_channels=3),
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # ])
    # train_root = r'/images/PublicDatasets/imagenet/train/'
    # train_dataset = torchvision.datasets.ImageFolder(train_root, transform=transform)
    # dataset_train_file = defaultValues['image_dir'] + 'dataset_train_160_grayscale.pickle'
    # with open(dataset_train_file, 'wb') as handle:
    #     pickle.dump(train_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # val_root = r'/images/PublicDatasets/imagenet_shared/val/'
    # val_dataset = torchvision.datasets.ImageFolder(val_root, transform=transform)
    # dataset_val_file = defaultValues['image_dir'] + 'dataset_val_160_grayscale.pickle'
    # with open(dataset_val_file, 'wb') as handle:
    #     pickle.dump(val_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(dataset_train_file, 'rb') as handle:
        dataset_train_new = pickle.load(handle)
    print(f'dataset_train_new {dataset_train_new}')
    with open(dataset_val_file, 'rb') as handle:
        dataset_val_new = pickle.load(handle)
    print(f'dataset_val_new {dataset_val_new}')
