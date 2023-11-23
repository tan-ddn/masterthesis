import torch
import json
from torchvision.transforms.functional import crop 
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTModel, ViTConfig
from transformers.modeling_outputs import ImageClassifierOutput
from dataset import SubDataset
from lightning_model_ViT import (
    fixation_ViTEmbeddings, conv_cat_ViTPatchEmbeddings,
    CosineWarmupScheduler, WarmupScheduler,
)

TRAIN_LABEL_FILE = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_train_split1.json'
VAL_LABEL_FILE = r'/images/innoretvision/cocosearch/coco_search18_labels_TP/coco_search18_fixations_TP_validation_split1.json'
IMAGE_DIR = r'/work/scratch/tnguyen/images/innoretvision/cocosearch/coco_search18_images_TP/'
MAX_FIX_LENGTH = 50

LR = 1e-2
AWD = 0.05
WARMUP_STEPS = 2400
LR_S = CosineWarmupScheduler  # 'CosineWarmupScheduler' or 'WarmupScheduler'

     
class fixation_ViT(ViTModel, torch.nn.Module):        
    def __init__(self, config: ViTConfig, 
                 add_pooling_layer: bool = True,
                 use_mask_token: bool = False,
                 patch_embeddings_type: str = 'conv_cat',
                 **kwargs) -> None:
        super().__init__(config, 
                         add_pooling_layer=add_pooling_layer, 
                         use_mask_token=use_mask_token)
        
        self.hparams = {
            'add_pooling_layer': add_pooling_layer,
            'use_mask_token': use_mask_token,
            'patch_embeddings_type': patch_embeddings_type,
        }
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


with open(TRAIN_LABEL_FILE) as file:
    train_data = json.load(file)
            
batch_size = 2
train_dataset = SubDataset(
    data=train_data,
    image_dir=IMAGE_DIR,
    max_fix_length=MAX_FIX_LENGTH,
    device='cpu',
)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=8,
)

classes = {
    "bottle": 0, "bowl": 1, "car": 2, "chair": 3, "clock": 4, "cup": 5,
    "fork": 6, "keyboard": 7, "knife": 8, "laptop": 9, "microwave": 10, "mouse": 11,
    "oven": 12, "potted plant": 13, "sink": 14, "stop sign": 15, "toilet": 16, "tv": 17,
}

configuration = ViTConfig(
    num_channels = 1,
    num_labels = 18,
    num_patches = MAX_FIX_LENGTH,
)
model = fixation_ViT(
    config = configuration,
    patch_embeddings_type = 'conv_cat',  # conv_cat, linear, or conv_linear
    lr = LR, awd = AWD, 
    warmup = WARMUP_STEPS, lr_s = LR_S,
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(2):
    running_loss = 0.0
    
    for i, data in enumerate(trainloader, 0):
        inputs, fixation_locations, labels = data[0].to(device), data[1].to(device), data[2].to(device)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=inputs, 
            fixation_locations=fixation_locations,
        ).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] Loss: {running_loss / 2000:.4f}')
            running_loss = 0.0

print(f'finished training')
