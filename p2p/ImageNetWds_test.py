import sys
import torch
import webdataset as wds
from torch.utils.data.sampler import Sampler
from itertools import islice
from tqdm import tqdm


sys.path.append("/home/students/tnguyen/masterthesis")

from dinov2_tan.data_transforms import make_classification_eval_transform, make_classification_train_transform


class ImageNetWds(wds.WebDataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.split = 'train'
        self.class_limit = 0

    def set_class_limit(self, limit):
        self.class_limit = limit

    def __len__(self):
        if self.split == 'train':
            return 1281167 if self.class_limit==0 else 1000*self.class_limit
        else:
            return 50000
        
    def __getitem__(self, index):
        return super().__getitem__(index)

    def set_split(self, split='train'):
        self.split = split
        return self
        
class ImageNetSubset(torch.utils.data.Dataset):
    def __init__(self, iterableData, class_limit, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        self.class_limit = class_limit
        self.data = []
        classes = [0] * 1000
        while sum(classes) < (1000*class_limit):
            image, label = next(iter(iterableData))
            if classes[label] < class_limit:
                self.data.append((image, label))
                classes[label] += 1
            # print(classes[label])
            # print(classes)
        print(sum(classes))
        print(len(self.data))

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]
        
class SubsetSampler(Sampler):
    def __init__(
        self,
        data_source, class_limit,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.class_limit = class_limit
        self.data = []
        self.classes = [0] * 1000

    def __len__(self):
        return 1000*self.class_limit
        
    def __iter__(self):
        # return iter(range(len(self.data)))
        image, label = next(iter(self.data_source))
        if self.classes[label] < self.class_limit:
            yield image, label
            self.classes[label] += 1
        print(self.classes)
        print(sum(self.classes))
        print(len(self.data))


def index_gen():
    for i in range(1281167):
        yield i

def get_train_dataset():
    train_transform = make_classification_train_transform(crop_size=224, grayscale=False, norm='norm')

    training_num_classes = 1000
    
    # train_data_dir = r'/images/innoretvision/eye/imagenet_patch/train/'
    train_data_dir = r'/images/innoretvision/eye/imagenet_patch/sub100_train/'
    # tar_range = r'000100..000110'
    # tar_range = r'000000..000010'
    tar_range = r'000000..000011'
    # tar_range = r'000000..000146'
    train_data_path = train_data_dir + 'imagenet-train-{' + tar_range + '}.tar'
    pil_dataset = (
        ImageNetWds(
        # wids.ShardListDataset(
            train_data_path,
        )
        # .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg", "cls")
    )

    def preprocess(sample):
        image, label = sample
        # image, label = sample[".jpg"], sample[".cls"]
        return train_transform(image), label

    train_dataset = pil_dataset#.map(preprocess)
    # train_dataset = ImageNetSubset(train_dataset, class_limit=2)

    return train_dataset

    # train_sampler = SubsetSampler(train_dataset, class_limit=2)

    # data_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     sampler=None,
    #     # sampler=train_sampler,
    #     batch_size=8,
    #     num_workers=1,
    #     pin_memory=True,
    # )

    # return data_loader

def main():
    train_data_loader = get_train_dataset()
    classes = [0] * 1000
    for _, label in tqdm(train_data_loader):
        classes[label] += 1
    print(classes)
    print(sum(classes))
    # for image, label in islice(train_data_loader, 0, 3):
    #     print(image, label)

if __name__ == '__main__':
    main()
