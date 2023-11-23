import json
import os
import logging
import time
import torch
# from torch.distributed.pipeline.sync import Pipe
from argparse import ArgumentParser
from pathlib import Path
from dataset import FixationDataset
from typing import Type
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from transformers import get_cosine_schedule_with_warmup

CONST = {
    'classes': 18,
    'channels': 1,
    'patch_size': (16, 16),
    'max_fix_length': 50,
    'data_limit': 500,
    'max_epochs': 2,
    'patience': 100,
    'min_delta': 0.00,  # For Early Stopping
    'result_dir': r'/home/students/tnguyen/results/',
    'train_label_file': r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json',
    'val_label_file': r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json',
    'image_dir': r'/work/scratch/tnguyen/images/cocosearch/patches/',
}

class TrainModel:
    def __init__(self, **kwargs):
        self.result_dir = CONST["result_dir"]
        self.channels = CONST["channels"]
        self.patch_size = CONST["patch_size"]
        self.max_fix_length = CONST["max_fix_length"]
        self.__dict__.update(kwargs)
        self.train_dataset = None
        self.val_dataset = None

        self.device, self.device_name, self.accelerator, self.num_devices = check_device()
        # self.device = 'cpu'

        self.args = set_args_from_cli()
        self.args.channels = int(self.args.channels)
        self.args.classes = int(self.args.classes)
        self.args.patience = int(self.args.patience)
        self.args.data_limit = int(self.args.data_limit)
        self.args.max_epochs = int(self.args.max_epochs)

    def set_output_dir(self):
        if self.model is None:
            raise ValueError("Model must not be None")
        self.output_dir = self.result_dir + self.model.__class__.__name__ + '/'

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

    def read_data(self, file_path):
        with open(file_path) as file:
            data = json.load(file)
        return data[0: self.args.data_limit]

    def setup_training(self):
        """Setup stage"""
        self.set_output_dir()
        self.set_logging()
        self.log_pid_and_gpu_info()
        self.train_logger.info(f"CLI args: {self.args}")

        """Get dataset"""
        if self.train_dataset is None:
            train_data = self.read_data(CONST['train_label_file'])
            train_dataset = FixationDataset(
                data=train_data,
                image_dir=CONST['image_dir'],
                max_fix_length=self.max_fix_length,
                channels=self.channels,
                patch_size=self.patch_size,
                device=self.device,
            )  
        else:
            train_dataset = self.train_dataset
        if self.val_dataset is None:
            val_data = self.read_data(CONST['val_label_file']) 
            val_dataset = FixationDataset(
                data=val_data,
                image_dir=CONST['image_dir'],
                max_fix_length=self.max_fix_length,
                channels=self.channels,
                patch_size=self.patch_size,
                device=self.device,
            )
        else:
            val_dataset = self.val_dataset

        self.trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_bs,
            shuffle=True,
            pin_memory=False,
            # num_workers=8,
        )
        self.valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.val_test_bs,
            shuffle=True,
            pin_memory=False,
            # num_workers=8,
        )
        self.train_logger.info(f"Total train data: {len(train_dataset)}")
        self.train_logger.info(f"Total valid data: {len(val_dataset)}")
        self.train_logger.info(f"Train batch size: {int(self.train_bs)}")
        self.train_logger.info(f"Valid batch size: {int(self.val_test_bs)}")

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def fit(self):
        """Train model"""
        """Initiate early stop"""

        """Save only model with the best val_acc metric"""

        """List of trainer callbacks"""


        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model.lr,
            weight_decay=self.model.weight_decay,
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        num_training_steps = self.args.max_epochs * len(self.trainloader)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.model.warmup,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(len(self.trainloader) + len(self.valloader)))

        """Training loop"""
        self.model.train()
        for epoch in range(self.args.max_epochs):
            progress_bar.reset()
            """Attributes for training metrics"""
            self.train_loss = torch.zeros(self.args.max_epochs, device=self.device)
            self.train_preds = torch.tensor([], device=self.device)
            self.train_targs = torch.tensor([], device=self.device)
            self.train_loss_single_step = 0
            self.train_step_length = 0
            self.val_loss = torch.zeros(self.args.max_epochs, device=self.device)
            self.val_preds = torch.tensor([], device=self.device)
            self.val_targs = torch.tensor([], device=self.device)
            self.val_loss_single_step = 0
            self.val_step_length = 0

            running_loss = 0.0
            
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = (
                    data[0].to(self.device), 
                    data[1].to(self.device),
                )

                optimizer.zero_grad()

                outputs = self.model(
                    pixel_values=inputs,
                ).logits
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                running_loss += loss.item()
                # if i % 2000 == 1999:
                #     print(f'[{epoch + 1}, {i + 1:5d}] Loss: {running_loss / 2000:.4f}')
                #     running_loss = 0.0
                progress_bar.update(1)
                
                self.train_step_length += 1
                self.train_loss_single_step += float(loss.item())
                self.train_preds = torch.cat((self.train_preds, outputs), 0)
                self.train_targs = torch.cat((self.train_targs, labels), 0)
            
            self.train_loss_single_step = self.train_loss_single_step / self.train_step_length
            # print(f"Average train loss {self.train_loss_single_step}")
            self.train_loss[epoch] = self.train_loss_single_step

            """Evaluation"""
            self.model.eval()
            for i, data in enumerate(self.valloader, 0):
                inputs, labels = (
                    data[0].to(self.device), 
                    data[1].to(self.device),
                )
                with torch.no_grad():
                    outputs = self.model(
                        pixel_values=inputs,
                    ).logits
                    loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                progress_bar.update(1)
                
                self.val_step_length += 1
                self.val_loss_single_step += float(loss.item())
                self.val_preds = torch.cat((self.val_preds, outputs), 0)
                self.val_targs = torch.cat((self.val_targs, labels), 0)
            
            self.val_loss_single_step = self.val_loss_single_step / self.val_step_length
            # print(f"Average val loss {self.val_loss_single_step}")
            self.val_loss[epoch] = self.val_loss_single_step

            """Calculate Acc"""
            predictions = self.val_preds
            pred_digits = torch.argmax(predictions, dim=1)
            pred_digits = pred_digits.cpu()
            targets = self.val_targs
            targets = targets.to(torch.int64).cpu()
            # print(f"predictions {predictions}")
            print(f"pred_digits {pred_digits}")
            conf_mat = confusion_matrix(targets, pred_digits)

            train_pred_digits = torch.argmax(self.train_preds, dim=1)
            train_pred_digits = train_pred_digits.cpu()
            train_targets = self.train_targs
            train_targets = train_targets.to(torch.int64).cpu()
            # print(f'train predictions: {train_pred_digits}')
            # print(f'train targets: {train_targets}')
            train_acc = balanced_accuracy_score(train_targets, train_pred_digits)
            val_acc = balanced_accuracy_score(targets, pred_digits)
            predictions = self.model.softmax(predictions.cpu())
            # print(f'Val preds {predictions}')
            val_auroc = roc_auc_score(targets, predictions, multi_class='ovr')

            """Log the outputs"""
            self.train_logger.info(f"\n")
            self.train_logger.info(f"Epoch: {epoch}")
            self.train_logger.info(f"current lr: {self.get_lr(optimizer)}")
            self.train_logger.info(f"train loss: {self.train_loss[epoch]}")
            self.train_logger.info(f"train acc: {train_acc}")
            self.train_logger.info(f"val loss: {self.val_loss[epoch]}")
            self.train_logger.info(f"val acc: {val_acc}")
            self.train_logger.info(f"val auroc: {val_auroc}")
            self.train_logger.info(f"Conf Matrix: {conf_mat}")

    # def test(self, ckpt_path=''):
    #     # self.trainer.test(ckpt_path=ckpt_path)
    #     self.trainer.test(model=self.model, datamodule=self.dataset)

    def run(self, model : Type[torch.nn.Module] = None, **kwargs):
        self.model = model

        self.setup_training()
        self.model.to(self.device)
        # if num_gpus > 1:
        #     self.model = Pipe(self.model.sequence, chunks=4)
        """Check the model's hparams"""
        self.train_logger.info(f"Hparams: {self.model.hparams_text}")
        self.train_logger.info(f'Model {self.model}')
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.train_logger.info(f'Trainable params {trainable_params}')
        self.train_logger.info(f'Total params {total_params}')

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
