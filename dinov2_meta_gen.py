from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/images/PublicDatasets/imagenet", extra="/images/PublicDatasets/imagenet")
    dataset.dump_extra()