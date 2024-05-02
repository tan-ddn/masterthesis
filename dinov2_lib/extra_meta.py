from dinov2.data.datasets import ImageNet


ROOT = r"/work/scratch/tnguyen/images/imagenette2/"
EXTRA = r"/work/scratch/tnguyen/images/imagenette2/"

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=ROOT, extra=EXTRA)
    dataset.dump_extra()
