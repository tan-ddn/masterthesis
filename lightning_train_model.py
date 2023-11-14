import os
import logging
import time

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
# from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from torch.distributed.pipeline.sync import Pipe
from argparse import ArgumentParser
from pathlib import Path
from dataset import DataModule
from typing import Type


CONST = {
    'data_limit': 500,
    'max_epochs': 2,
    'patience': 100,
    'channels': 1,
    'classes': 18,
    'min_delta': 0.00,  # For Early Stopping
    'result_dir': r'/home/students/tnguyen/results/'
}

class TrainModel:
    def __init__(self, modelClass : Type[pl.LightningModule],
                 **kwargs):
        self.result_dir = CONST["result_dir"]
        self.__dict__.update(kwargs)
        self.modelClass = modelClass

        self.device, self.device_name, self.accelerator, self.num_devices = check_device()

        self.args = set_args_from_cli()
        self.args.channels = int(self.args.channels)
        self.args.classes = int(self.args.classes)
        self.args.patience = int(self.args.patience)
        self.args.data_limit = int(self.args.data_limit)
        self.args.max_epochs = int(self.args.max_epochs)

    def set_output_dir(self):
        if self.modelClass is None:
            raise ValueError("Model must not be None")
        self.output_dir = self.result_dir + self.modelClass.__name__ + '/'

    def set_logging(self, log_filename="log_" + str(int(time.time()))):
        self.log_filename = log_filename

        """Create an output directory if it doesn't exist"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        """Set up logging system"""
        log_file = self.output_dir + self.log_filename + ".txt"

        format_str = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        formatter = logging.Formatter(format_str)

        """Set up file log"""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        """Set up console log"""
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)

        """Configure logging for model file"""
        self.train_logger = logging.getLogger("pl_train")
        self.train_logger.setLevel(logging.DEBUG)
        self.train_logger.addHandler(file_handler)
        self.train_logger.addHandler(console)
        self.train_logger.info(f"Train log is set up")

        """Configure logging at the root level of Lightning"""
        self.lightning_logger = logging.getLogger("pytorch_lightning")
        # remove the console handler
        for hdlr in self.lightning_logger.handlers[:]:
            if isinstance(hdlr, logging.StreamHandler):
                self.lightning_logger.removeHandler(hdlr)
        self.lightning_logger.addHandler(file_handler)
        self.lightning_logger.addHandler(console)
        self.lightning_logger.info(f"Lightning log is set up")

    def log_pid_and_gpu_info(self):
        self.train_logger.info(f'PID: {os.getpid()}')  # Process ID
        print_gpu_info()

    def setup_training(self):
        """Setup stage"""
        self.set_output_dir()
        self.set_logging()
        self.log_pid_and_gpu_info()
        self.train_logger.info(f"CLI args: {self.args}")

        """Set seed for reproducibility"""
        pl.seed_everything(42, workers=True)

        """Get dataset"""
        self.dataset = DataModule(
            train_bs=self.train_bs,
            val_test_bs=self.val_test_bs,
            max_fix_length=self.max_fix_length,
            data_limit=self.args.data_limit,
            # device=self.device,
            device='cpu',
            logger=self.train_logger
        )

    def fit(self):
        """Train model"""
        # self.TB_logger = TensorBoardLogger(self.output_dir)

        """Initiate early stop"""
        early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=CONST["min_delta"],
            patience=self.args.patience,
            verbose=False,
            mode="max",
            check_on_train_epoch_end=True,
        )

        """Save only model with the best val_acc metric"""
        save_chckpt_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_acc",
            mode="max",
            filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
            save_on_train_epoch_end=True,
        )

        """List of trainer callbacks"""
        trainer_callbacks = [save_chckpt_callback, early_stop_callback]

        """Version name with or without data augmentation"""
        self.csv_logger = CSVLogger(self.output_dir, version=self.log_filename)

        self.trainer = pl.Trainer(
            logger=self.csv_logger,
            deterministic=True,
            default_root_dir=self.output_dir,
            max_epochs=self.args.max_epochs,
            accelerator=self.accelerator,
            devices=self.num_devices,
            callbacks=trainer_callbacks,)
        self.trainer.fit(model=self.model, datamodule=self.dataset)

    # def test(self, ckpt_path=''):
    #     # self.trainer.test(ckpt_path=ckpt_path)
    #     self.trainer.test(model=self.model, datamodule=self.dataset)

    def run(self, **kwargs):
        self.setup_training()

        self.model = self.modelClass(
            **kwargs
        )
        # if num_gpus > 1:
        #     self.model = Pipe(self.model.sequence, chunks=4)
        self.model.prep()
        self.train_logger.info(f'Model {self.model}')

        self.fit()


def check_device():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        accelerator = 'gpu'
        num_devices = -1
    else:
        device = 'cpu'
        device_name = 'cpu'
        accelerator = 'cpu'
        num_devices = 1
    return device, device_name, accelerator, num_devices

def print_gpu_info():
    if torch.cuda.is_available():
        print(f"Device count {torch.cuda.device_count()}")
        print(f"Current device {torch.cuda.current_device()}")
        print(f"Device name {torch.cuda.get_device_name(torch.cuda.current_device())}")

def set_args_from_cli():
    parser = ArgumentParser()

    """Train arguments"""
    parser.add_argument("-ch", "--channels", dest="channels", type=str, default=str(CONST["channels"]))
    parser.add_argument("-cl", "--classes", dest="classes", type=str, default=CONST["classes"])

    parser.add_argument("-me", "--max_epochs", dest="max_epochs", type=str, default=str(CONST["max_epochs"]))  # Number of max epochs per training
    parser.add_argument("-dl", "--data_limit", dest="data_limit", type=str, default=str(CONST["data_limit"]))  # Number of data points
    parser.add_argument("-pa", "--patience", dest="patience", type=str, default=str(CONST["patience"]))  # PATIENCE for early stop

    """Parse the user inputs and defaults"""
    args = parser.parse_args()
    # print(args)
    return args
