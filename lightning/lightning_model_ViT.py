import logging
import torch
import numpy as np
from torchvision.transforms.functional import crop 
from lightning.pytorch import LightningModule
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from transformers.modeling_outputs import ImageClassifierOutput
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from lightning_train_model import TrainModel


class linear_ViTPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        
        in_features = self.num_channels * self.patch_size[0] * self.patch_size[1]
        self.projection = torch.nn.Linear(in_features, config.hidden_size)
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_patches, num_channels, height, width = pixel_values.shape
        
        pixel_values = pixel_values.flatten(2)
        embeddings = self.projection(pixel_values).transpose(1, 2)  # Shape: (batch_size, hidden_size, num_patches)
        return embeddings

class conv_linear_ViTPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self, config):
        super().__init__(config)

        self.projection = torch.nn.Sequential(
            torch.nn.Conv2d(self.num_channels, config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size),
            torch.nn.Flatten(start_dim=2),
            torch.nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_patches, num_channels, height, width = pixel_values.shape
        
        embeddings = []
        for i in range(batch_size):
            one_item_embeddings = self.projection(pixel_values[i])
            one_item_embeddings = torch.squeeze(one_item_embeddings)  # Shape: (num_patches, hidden_size)
            embeddings.append(one_item_embeddings)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings

class conv_cat_ViTPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self, config):
        super().__init__(config)
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_patches, num_channels, height, width = pixel_values.shape
        
        embeddings = []
        for i in range(batch_size):
            one_item_embeddings = self.projection(pixel_values[i])
            one_item_embeddings = torch.squeeze(one_item_embeddings)  # Shape: (num_patches, hidden_size)
            embeddings.append(one_item_embeddings)
        embeddings = torch.stack(embeddings, dim=0)
        # print(f'Embeddings shape {embeddings.shape}')
        return embeddings

class fixation_ViTEmbeddings(ViTEmbeddings):
    def __init__(self, config: ViTConfig, 
                 use_mask_token: bool = False,
                 patch_embeddings_type: str = 'conv_cat') -> None:
        super().__init__(config, use_mask_token)
        
        self.patch_embeddings = conv_cat_ViTPatchEmbeddings(config)
        if patch_embeddings_type == 'linear':
            self.patch_embeddings = linear_ViTPatchEmbeddings(config)
        if patch_embeddings_type == 'conv_linear':
            self.patch_embeddings = conv_linear_ViTPatchEmbeddings(config)
        num_patches = config.num_patches
        self.position_embeddings = torch.nn.Parameter(
            torch.randn(1, num_patches + 1, config.hidden_size)
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_fixations, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        # print(f'Embeddings shape {embeddings.shape}')
        # print(f'Position embeddings shape {self.position_embeddings.shape}')
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

# define the ViT
class fixation_ViT(ViTModel, LightningModule):     
    def __init__(self, config: ViTConfig, 
                 add_pooling_layer: bool = True,
                 use_mask_token: bool = False,
                 patch_embeddings_type: str = 'conv_cat',
                 **kwargs) -> None:
        super().__init__(config, 
                         add_pooling_layer=add_pooling_layer, 
                         use_mask_token=use_mask_token)
        
        """Save hyperparams to self.hparams attribute"""
        self.save_hyperparameters()

        self.hparams['add_pooling_layer'] = add_pooling_layer
        self.hparams['use_mask_token'] = use_mask_token
        self.hparams['patch_embeddings_type'] = patch_embeddings_type
        """Convert config attrs and hyperparam values to text to save to log file"""
        args_lst = []
        attrs = filter(lambda attr: not attr.startswith('__'), dir(config))
        for key in attrs:
            value = getattr(config, key)
            if isinstance(value, float):
                value = str(key) + f"{value:0.1e}"
            else:
                value = str(key) + str(value)
            args_lst.append(value)
        for key, value in self.hparams.items():
            if isinstance(value, float):
                value = str(key) + f"{value:0.1e}"
            else:
                value = str(key) + str(value)
            args_lst.append(value)
        self.hparams_text = "_".join(args_lst)
        print(f'Config {config}')

        self.num_labels = config.num_labels
        self.patch_size = config.patch_size
        self.embeddings = fixation_ViTEmbeddings(
            config, 
            use_mask_token=use_mask_token,
            patch_embeddings_type=patch_embeddings_type,
        )
        # Classifier head
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else torch.nn.Identity()

        self.softmax = torch.nn.Softmax(dim=1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        fixation_locations: Optional[list] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pixel_values = self.batch_image2fixations(
            pixel_values=pixel_values, 
            fixation_locations=fixation_locations
        )

        outputs = super().forward(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    def prep(self):

        """Get Logger"""
        self.train_logger = logging.getLogger("pl_train")
        self.train_logger.debug(f"Logging is working")

        """Check the model's hparams"""
        self.train_logger.info(f"Hparams: {self.hparams_text}")

        self.loss_fn = torch.nn.CrossEntropyLoss()
        # self.mcauroc = MulticlassAUROC(num_classes=self.num_classes, average="macro", thresholds=5)
        # self.train_mca_avg = MulticlassAccuracy(num_classes=self.num_classes, average='micro')
        # self.mca_avg = MulticlassAccuracy(num_classes=self.num_classes, average='micro')
        # self.mcconfmat = MulticlassConfusionMatrix(num_classes=self.num_classes)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr, 
            weight_decay=self.hparams.awd
        )
        self.configure_lr_scheduler()
        return self.optimizer

    def configure_lr_scheduler(self):
        # print(f"Max epoch {self.trainer.max_epochs}")
        self.lr_scheduler = None
        if self.hparams.lr_s.__name__ == 'WarmupScheduler':
            self.lr_scheduler = WarmupScheduler(
                self.optimizer, warmup=self.hparams.warmup, d_model=self.d_model,
            )
        if self.hparams.lr_s.__name__ == 'CosineWarmupScheduler':
            self.train_logger.info(f"estimated_stepping_batches {self.trainer.estimated_stepping_batches}")
            # max_steps = self.trainer.max_epochs*self.trainer.num_training_batches
            self.lr_scheduler = CosineWarmupScheduler(
                self.optimizer, warmup=self.hparams.warmup,
                max_steps=self.trainer.estimated_stepping_batches,
            )

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()  # Step per iteration

    def on_fit_start(self):
        self.plot_values = []
        """Attributes for training metrics"""
        self.register_buffer("train_loss", torch.zeros(self.trainer.max_epochs, device=self.device))
        self.register_buffer("val_loss", torch.zeros(self.trainer.max_epochs, device=self.device))
        self.register_buffer("train_preds", torch.tensor([], device=self.device))
        self.register_buffer("train_targs", torch.tensor([], device=self.device))
        self.register_buffer("val_preds", torch.tensor([], device=self.device))
        self.register_buffer("val_targs", torch.tensor([], device=self.device))

        self.train_loss_single_step = 0
        self.train_step_length = 0
        self.val_loss_single_step = 0
        self.val_step_length = 0

        """Log hyperparams to Lightning logger"""
        self.logger.log_hyperparams(self.hparams)

    def on_train_epoch_start(self):
        """Get the current learning rate"""
        lightning_optimizer = self.optimizers()  # self = your model
        for pg in lightning_optimizer.optimizer.param_groups:
            self.train_logger.info(f"learning rate: {pg['lr']}")

    def forward_step(self, batch, batch_idx):
        input, fixation_locations, target = batch
        prediction = self(
            pixel_values=input, 
            fixation_locations=fixation_locations
        ).logits
        # print(f"prediction shape: {prediction.shape}")
        # print(f"target shape: {target.shape}")
        loss = self.loss_fn(prediction, target)

        return loss, prediction, target

    def training_step(self, batch, batch_idx):
        """training_step defines the train loop"""
        loss, prediction, target = self.forward_step(batch, batch_idx)

        # self.train_mca_avg(prediction, target)
        # self.train_preds += prediction
        # self.train_targs += target
        self.train_step_length += 1
        self.train_loss_single_step += float(loss.item())
        self.train_preds = torch.cat((self.train_preds, prediction), 0)
        self.train_targs = torch.cat((self.train_targs, target), 0)

        # print(f"\nTraining loss {loss}")

    def validation_step(self, batch, batch_idx):
        loss, prediction, target = self.forward_step(batch, batch_idx)

        # self.mca_avg(prediction, target)
        # self.mcauroc(prediction, target)

        # print(f"\nValidation loss {loss}")
        self.val_step_length += 1
        self.val_loss_single_step += float(loss.item())
        # print(f"Accumulative val loss {val_loss}")
        self.val_preds = torch.cat((self.val_preds, prediction), 0)
        self.val_targs = torch.cat((self.val_targs, target), 0)

    def on_validation_epoch_end(self):
        """No need to save metrics when in Sanity Checking stage"""
        if not self.trainer.sanity_checking:
            self.val_loss_single_step = self.val_loss_single_step / self.val_step_length
            # print(f"Average val loss {self.val_loss_single_step}")
            self.val_loss[self.current_epoch] = self.val_loss_single_step

    def on_train_epoch_end(self):
        self.train_loss_single_step = self.train_loss_single_step / self.val_step_length
        # print(f"Average train loss {self.train_loss_single_step}")
        self.train_loss[self.current_epoch] = self.train_loss_single_step

        """Calculate Acc"""
        predictions = self.val_preds
        pred_digits = torch.argmax(predictions, dim=1)
        pred_digits = pred_digits.cpu()
        targets = self.val_targs
        targets = targets.to(torch.int64).cpu()
        # print(f"predictions {predictions}")
        print(f"pred_digits {pred_digits}")
        # mcauroc = self.mcauroc(predictions, targets).item()
        # mca_avg = self.mca_avg(predictions, targets).item()
        # conf_mat = self.mcconfmat(predictions, targets)
        conf_mat = confusion_matrix(targets, pred_digits)

        # self.train_mca_avg(predictions, targets)
        # self.mca_avg(predictions, targets)
        # self.mcauroc(predictions, targets)
        # train_acc = self.train_mca_avg.compute()
        # val_acc = self.mca_avg.compute()
        # val_auroc = self.mcauroc.compute()
        train_pred_digits = torch.argmax(self.train_preds, dim=1)
        train_pred_digits = train_pred_digits.cpu()
        train_targets = self.train_targs
        train_targets = train_targets.to(torch.int64).cpu()
        # print(f'train predictions: {train_pred_digits}')
        # print(f'train targets: {train_targets}')
        train_acc = balanced_accuracy_score(train_targets, train_pred_digits)
        val_acc = balanced_accuracy_score(targets, pred_digits)
        predictions = self.softmax(predictions.cpu())
        # print(f'Val preds {predictions}')
        val_auroc = roc_auc_score(targets, predictions, multi_class='ovr')
        # try:
        #     val_auroc = roc_auc_score(pred_digits, targets, multi_class='ovr')
        # except ValueError:
        #     val_auroc = 0.0

        """Log the outputs"""
        self.train_logger.info(f"\n")
        self.train_logger.info(f"Epoch: {self.current_epoch}")
        self.train_logger.info(f"train loss: {self.train_loss[self.current_epoch]}")
        self.train_logger.info(f"train acc: {train_acc}")
        self.train_logger.info(f"val loss: {self.val_loss[self.current_epoch]}")
        self.train_logger.info(f"val acc: {val_acc}")
        self.train_logger.info(f"val auroc: {val_auroc}")
        self.train_logger.info(f"Conf Matrix: {conf_mat}")
        self.log_dict({
            'train_loss': self.train_loss[self.current_epoch],
            'train_acc': train_acc,
            'val_loss': self.val_loss[self.current_epoch],
            'val_acc': val_acc,
            'val_auroc': val_auroc,
            'step': float(self.current_epoch + 1),
        })

        """Reset validation predictions and targets for each epoch"""
        self.train_preds = torch.tensor([], device=self.device)
        self.train_targs = torch.tensor([], device=self.device)
        self.val_preds = torch.tensor([], device=self.device)
        self.val_targs = torch.tensor([], device=self.device)

        self.train_loss_single_step = 0
        self.train_step_length = 0
        self.val_loss_single_step = 0
        self.val_step_length = 0

    def batch_image2fixations(self, 
                              pixel_values : torch.Tensor, 
                              fixation_locations : torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = pixel_values.shape
        if type(self.patch_size) is tuple:
            patch_size_x = self.patch_size[0]
            patch_size_y = self.patch_size[1]
            margin_x = - patch_size_x // 2
            margin_y = - patch_size_y // 2
        else:
            patch_size_x = patch_size_y = self.patch_size
            margin_x = margin_y = - patch_size_x // 2
        # print(f'Fixation locations shape {fixation_locations.shape}')
        batch_size, num_patches, _ = fixation_locations.shape
        fixations = torch.zeros(
            (batch_size, num_patches, self.config.num_channels, 
             patch_size_y, patch_size_x), 
            device=self.device
        )
        for i, single_img in enumerate(pixel_values): 
            for j, item in enumerate(fixation_locations[i]):
                x = int(item[0])
                y = int(item[1])
                # print((x, y))
                if x < 0 or y < 0:
                    continue
                top = int(y + margin_y)
                left = int(x + margin_x)
                height = int(patch_size_y)
                width = int(patch_size_x)
                fixations[i][j] = crop(
                    single_img, top=top, left=left, 
                    height=height, width=width
                )
        return fixations


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, d_model):
        self.warmup = warmup
        self.d_model = d_model
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self._step_count)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        lr_factor = 1e3 * self.d_model**(-0.5) * min(step**(-0.5), step * self.warmup**(-1.5))
        return lr_factor

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_steps):
        self.warmup = warmup
        self.max_steps = max_steps
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self._step_count)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_steps))
        if step <= self.warmup:
            lr_factor *= step * 1.0 / self.warmup
        # print(f"step {step}")
        # print(f"max_steps {self.max_steps}")
        # print(f"lr_factor {lr_factor}")
        # print(f"warmup {self.warmup}")
        return lr_factor
    

def main():
    MAX_FIX_LENGTH = 50
    LR = 1e-2
    AWD = 0.05
    WARMUP_STEPS = 2400
    LR_S = CosineWarmupScheduler  # 'CosineWarmupScheduler' or 'WarmupScheduler'

    training = TrainModel(modelClass = fixation_ViT,
                          train_bs = 2, val_test_bs = 2,
                          max_fix_length = MAX_FIX_LENGTH,)
    configuration = ViTConfig(
        num_channels = training.args.channels,
        num_labels = training.args.classes,
        num_patches = MAX_FIX_LENGTH,
    )
    training.run(
        config = configuration,
        patch_embeddings_type = 'conv_cat',  # conv_cat, linear, or conv_linear
        lr = LR, awd = AWD, 
        warmup = WARMUP_STEPS, lr_s = LR_S,
    )

if __name__ == "__main__":
    main()
