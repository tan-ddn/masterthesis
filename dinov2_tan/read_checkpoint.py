import torch

chkp_file = '/work/scratch/tnguyen/dinov2/fixation/45/model_final.pth'
state_dict = torch.load(chkp_file, map_location="cpu")
for entity, _ in state_dict.items():
    print(f"Entity {entity}")
    for k, v in state_dict[entity].items():
        print(f"Key {k}")

# Key classifiers_dict.classifier_1_blocks_avgpool_False_lr_0_00007.linear.weight
# Key classifiers_dict.classifier_1_blocks_avgpool_False_lr_0_00007.linear.bias

# classifier_4_blocks_avgpool_False_lr_0_32500
