import os
import torch
from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download




def uni(weight):
    # login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens


    model = timm.create_model(
    # "vit_large_patch16_224", img_size=224, init_values=1e-5, num_classes=0, dynamic_img_size=True
    "vit_large_patch16_224", img_size=224, init_values=1e-5, num_classes=0
)
    model.load_state_dict(torch.load(weight, map_location="cpu"), strict=True)
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ]
    # )
    model.eval()

    return model