import os
import torch
import numpy as np
from argparse import ArgumentParser
from tqdm.auto import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from utils.logging_utils import Log


TRAIN_CONST = {
    'data_dir': r'/images/PublicDatasets/imagenet',
    'data_limit': 500,
    'num_class': 18,
    'num_channel': 1,
    'patch_size': (16, 16),
    'max_fix_length': 50,
    'max_epochs': 2,
    'train_batch_size': 128,
    'val_batch_size': 256,
    'lr': 0.001,
    'patience': 100,
    'min_delta': 0.00,  # For Early Stopping
    'result_dir': r'/home/students/tnguyen/results/',
    'train_label_file': r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json',
    'val_label_file': r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json',
    'image_dir': r'/work/scratch/tnguyen/images/cocosearch/patches/',
}

def set_args_from_cli(defaultValues : dict = None):
    """Parse args from cli"""
    if defaultValues == None:
        defaultValues = TRAIN_CONST

    parser = ArgumentParser(description="PyTorch training")

    parser.add_argument("-dd", "--data-dir", dest="data_dir", type=str, default=defaultValues['data_dir'])
    parser.add_argument("-dl", "--data_limit", dest="data_limit", type=str, default=str(defaultValues["data_limit"]))
    parser.add_argument("-nch", "--num_channel", dest="num_channel", type=str, default=str(defaultValues["num_channel"]))
    parser.add_argument("-ncl", "--num_class", dest="num_class", type=str, default=defaultValues["num_class"])
    parser.add_argument("-me", "--max_epochs", dest="max_epochs", type=str, default=str(defaultValues["max_epochs"])) 
    parser.add_argument("-tbs", "--train_batch_size", dest="train_batch_size", type=str, default=str(defaultValues["train_batch_size"])) 
    parser.add_argument("-vbs", "--val_batch_size", dest="val_batch_size", type=str, default=str(defaultValues["val_batch_size"])) 
    parser.add_argument("-lr", "--lr", dest="lr", type=str, default=str(defaultValues["lr"])) 
    parser.add_argument("-pa", "--patience", dest="patience", type=str, default=str(defaultValues["patience"]))

    """Parse the user inputs and defaults"""
    args = parser.parse_args()
    # print(args)
    return args

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


class TrainHuggingFace():
    def __init__(
            self, args, model, loss_fn, optimizer, scheduler,
            trainloader, valloader, dataset_sizes, result_dir
        ) -> None:
        self.device, self.device_name, self.accelerator, self.num_devices = check_device()

        self.args = args
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.valloader = valloader
        self.dataset_sizes = dataset_sizes
        self.progress_bar = tqdm(range(len(trainloader) + len(valloader)))
        self.output_dir = result_dir + self.model.__class__.__name__ + '/'
        self.log = Log(self.output_dir)
        self.log.logger.info(f'PID: {os.getpid()}')
        self.log.logger.info(f"CLI args: {self.args}")
        print_gpu_info()
        
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    def fit_loop(self) -> None:
        """Attributes for training metrics"""
        running_loss = 0.0
        predictions = np.array([])
        targets = np.array([])
        for i, data in enumerate(self.trainloader, 0):
            inputs, labels = (
                data[0].to(self.device), 
                data[1].to(self.device),
            )

            # forward
            if self.phase == 'train':
                # zero the parameter gradients
                self.optimizer.zero_grad()
                outputs = self.model(
                    pixel_values=inputs,
                ).logits
            else:
                with torch.no_grad():
                    outputs = self.model(
                        pixel_values=inputs,
                    ).logits
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)

            # backward + optimize only if in training phase
            if self.phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            running_loss += float(loss.item())
            predictions.append(preds)
            targets.append(labels)
            self.progress_bar.update(1)
        return outputs, running_loss, predictions, targets

    def train_model(self):   
        for epoch in range(self.args.max_epochs):
            self.progress_bar.reset()
            """Training loop"""
            self.phase = 'train'
            self.model.train()         
            outputs, loss, predictions, targets = self.fit_loop()
            
            loss = loss / self.dataset_sizes['train']
            accuracy = balanced_accuracy_score(targets, predictions)
            self.log.logger.info(f"\n")
            self.log.logger.info(f"Epoch: {epoch}")
            self.log.logger.info(f"current lr: {self.get_lr(self.optimizer)}")
            self.log.logger.info(f"train loss: {loss}")
            self.log.logger.info(f"train acc: {accuracy}")

            """Evaluation"""
            self.phase = 'eval'
            self.model.eval()
            outputs, loss, predictions, targets = self.fit_loop()
            
            loss = loss / self.dataset_sizes['val']
            accuracy = balanced_accuracy_score(targets, predictions)
            conf_mat = confusion_matrix(targets, predictions)
            softmax_outputs = torch.nn.functional.softmax(outputs)
            val_auroc = roc_auc_score(targets, softmax_outputs, multi_class='ovr')
            """Log the outputs"""
            self.train_logger.info(f"val loss: {loss}")
            self.train_logger.info(f"val acc: {accuracy}")
            self.train_logger.info(f"val auroc: {val_auroc}")
            self.train_logger.info(f"Conf Matrix: {conf_mat}")
        