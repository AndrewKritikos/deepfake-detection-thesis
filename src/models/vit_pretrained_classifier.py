import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class PretrainedVit(nn.Module):
    def __init__(self, num_classes: int = 2, freeze_base: bool = True):
        """
        Freeze base, freezes the internal weights of the model and trains only the last layer
        for less computational power
        """

        super().__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights)

        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)