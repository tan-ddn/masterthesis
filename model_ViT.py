import torch
import numpy as np
from torchvision.transforms.functional import crop
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTConfig, ViTModel
from transformers.models.vit.modeling_vit import ViTPatchEmbeddings, ViTEmbeddings
from transformers.modeling_outputs import ImageClassifierOutput
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from utils.train_utils import *
from dataset_coco18tp import coco18tp_data
from transformers import get_cosine_schedule_with_warmup


class linear_ViTPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.num_patches = config.num_patches

        in_features = self.num_channels * self.patch_size[0] * self.patch_size[1]
        self.projection = torch.nn.Linear(in_features, config.hidden_size)
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_patches, num_channels, height, width = pixel_values.shape
        
        pixel_values = pixel_values.flatten(2)
        embeddings = self.projection(pixel_values)  # Shape: (batch_size, num_patches, hidden_size)
        return embeddings

class conv_linear_ViTPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self, config,):
        super().__init__(config)
        self.num_patches = config.num_patches

        self.projection = torch.nn.Conv2d(self.num_channels, config.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
    
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_patches, num_channels, height, width = pixel_values.shape
        
        embeddings = []
        for i in range(batch_size):
            # print(f'pixel value shape {pixel_values[i].shape}')
            one_item_embeddings = self.projection(pixel_values[i]).flatten(1)
            # print(f'after conv2d shape {one_item_embeddings.shape}')
            one_item_embeddings = self.linear(one_item_embeddings)
            one_item_embeddings = torch.squeeze(one_item_embeddings)  # Shape: (num_patches, hidden_size)
            embeddings.append(one_item_embeddings)
        embeddings = torch.stack(embeddings, dim=0)
        return embeddings

class conv_cat_ViTPatchEmbeddings(ViTPatchEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.num_patches = config.num_patches
    
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
                 patch_embeddings_type: str = '') -> None:
        super().__init__(config, use_mask_token)
        
        self.patch_embeddings_type = patch_embeddings_type
        self.patch_embeddings = ViTPatchEmbeddings(config)
        if patch_embeddings_type == 'linear':
            self.patch_embeddings = linear_ViTPatchEmbeddings(config)
        if patch_embeddings_type == 'conv_linear':
            self.patch_embeddings = conv_linear_ViTPatchEmbeddings(config)
        if patch_embeddings_type == 'conv_cat':
            self.patch_embeddings = conv_cat_ViTPatchEmbeddings(config)
        if patch_embeddings_type != '':
            num_patches = self.patch_embeddings.num_patches
            self.position_embeddings = torch.nn.Parameter(
                torch.randn(1, num_patches + 1, config.hidden_size)
            )

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        if self.patch_embeddings_type == '':
            batch_size, num_channels, height, width = pixel_values.shape
        else:
            batch_size, num_fixations, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding,)

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
class fixation_ViT(ViTModel, torch.nn.Module):        
    def __init__(self, config: ViTConfig, 
                 add_pooling_layer: bool = True,
                 use_mask_token: bool = False,
                 patch_embeddings_type: str = '',
                 **kwargs) -> None:
        super().__init__(config, 
                         add_pooling_layer=add_pooling_layer, 
                         use_mask_token=use_mask_token)
        self.__dict__.update(kwargs)
        
        kwargs['add_pooling_layer'] = add_pooling_layer
        kwargs['use_mask_token'] = use_mask_token
        kwargs['patch_embeddings_type'] = patch_embeddings_type
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
        for key, value in kwargs.items():
            if isinstance(value, float):
                value = str(key) + f"{value:0.1e}"
            else:
                value = str(key) + str(value)
            args_lst.append(value)
        self.hparams_text = "__".join(args_lst)
        # print(f'Config {config}')

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

def coco18tp():
    WEIGHT_DECAY = 0.05
    WARMUP_STEPS = 7000
    LR_S = None  # 'CosineWarmupScheduler' or 'WarmupScheduler'

    defaultValues = TRAIN_CONST
    defaultValues['lr'] = 1e-7
    defaultValues['train_batch_size'] = 256
    defaultValues['val_batch_size'] = 512
    args = set_args_from_cli(defaultValues)
    
    configuration = ViTConfig(
        num_channels = args.num_channel,
        num_labels = args.num_class,
        num_patches = defaultValues['max_fix_length'],
    )
    model = fixation_ViT(
        config = configuration,
        patch_embeddings_type = 'conv_cat',  # '', conv_cat, linear, or conv_linear
    )
    trainloader, valloader, dataset_sizes = coco18tp_data(args, defaultValues)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = args.lr, 
        weight_decay = WEIGHT_DECAY, 
    )
    num_training_steps = args.max_epochs * len(trainloader)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )
    
    training = TrainHuggingFace(
        args, model, loss_fn, optimizer,lr_scheduler,
        trainloader, valloader, dataset_sizes, 
        defaultValues['result_dir'],
    )
    training.train_model()

def pretrained_and_imagenet():
    import torchvision.transforms as transforms 
    from datasets import load_dataset
    from transformers import AutoImageProcessor
    from tqdm.auto import tqdm
    from dataset_imagenet import ImageNetData

    WEIGHT_DECAY = 0.05
    WARMUP_STEPS = 7000
    LR_S = None  # 'CosineWarmupScheduler' or 'WarmupScheduler'

    defaultValues = TRAIN_CONST
    defaultValues['num_class'] = 1000
    defaultValues['num_channel'] = 3
    defaultValues['lr'] = 1e-5
    defaultValues['train_batch_size'] = 256
    defaultValues['val_batch_size'] = 512
    defaultValues['data_dir'] = r'/images/PublicDatasets/imagenet'
    args = set_args_from_cli(defaultValues)
    
    model = fixation_ViT.from_pretrained(
        "masterthesis/huggingface/vit-base-patch16-224",
        num_labels = args.num_class,
        patch_embeddings_type = '',  # '', conv_cat, linear, or conv_linear
    )

    dataloaders, dataset_sizes = ImageNetData(
        args.data_dir, 
        args.data_limit,
        return_dataset=False
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = args.lr, 
        weight_decay = WEIGHT_DECAY, 
    )
    num_training_steps = args.max_epochs * len(dataloaders['train'])
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=num_training_steps,
    )

    training = TrainHuggingFace(
        args, model, loss_fn, optimizer,lr_scheduler,
        dataloaders['train'], dataloaders['val'], dataset_sizes, 
        defaultValues['result_dir'],
    )
    training.train_model()


if __name__ == "__main__":
    coco18tp()
    # pretrained_and_imagenet()
