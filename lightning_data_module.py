import json
import torch
import lightning.pytorch as pl
from dataset import TRAIN_LABEL_FILE, VAL_LABEL_FILE, IMAGE_DIR, SubDataset

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_label_file : str = TRAIN_LABEL_FILE,
                 val_label_file : str = VAL_LABEL_FILE,
                 image_dir : str = IMAGE_DIR,
                 train_bs : int = 8,
                 val_test_bs : int = 8,
                 max_fix_length : int = 0,
                 data_limit : int = -1,
                 device : str = 'cpu',
                 logger : None = None):
        """
        :param train_label_file:
        :param val_label_file:
        :param image_dir:
        :param train_bs:
        :param val_test_bs:
        :param data_limit:
        :param device:
        :param logger:
        """
        super().__init__()
        self.train_label_file = train_label_file
        self.val_label_file = val_label_file
        self.image_dir = image_dir
        self.train_bs = train_bs
        self.val_test_bs = val_test_bs
        self.max_fix_length = max_fix_length
        self.logger = logger
        self.device = device

        with open(train_label_file) as file:
            self.train_data = json.load(file)
        with open(val_label_file) as file:
            self.val_data = json.load(file)

        # print(train_label_file)
        if data_limit > 0:
            self.train_data = self.train_data[0:data_limit]
            self.val_data = self.val_data[0:data_limit]
            # self.test_data = self.test_data[0:data_limit]

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = SubDataset(
                data=self.train_data,
                image_dir=self.image_dir,
                max_fix_length=self.max_fix_length,
                device=self.device,
                )
            self.val_dataset = SubDataset(
                data=self.val_data,
                image_dir=self.image_dir,
                max_fix_length=self.max_fix_length,
                device=self.device,
                )
            # self.test_dataset = SubDataset(
            #     data=self.test_data,
            #     image_dir=self.image_dir,
            #     max_fix_length=self.max_fix_length,
            #     device=self.device,
            #     )

        self.logger.info(f"Total train data: {len(self.train_dataset)}")
        self.logger.info(f"Total valid data: {len(self.val_dataset)}")

        self.logger.info(f"Train batch size: {int(self.train_bs)}")
        self.logger.info(f"Valid batch size: {int(self.val_test_bs)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_bs,
            shuffle=False,
            pin_memory=False,
            num_workers=8,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.val_test_bs,
            shuffle=False,
            pin_memory=False,
        )

    def test_dataloader(self):
        if self.test_dataset is None or len(self.test_dataset) == 0:
            return []
        # return torch.utils.data.DataLoader(self.test_dataset,
        #                   batch_size=self.val_test_bs,
        #                   shuffle=False,
        #                   num_workers=self.num_workers,
        #                   pin_memory=False)


def main():
    data = DataModule()
    data.setup(stage='fit')

if __name__ == "__main__":
    main()
