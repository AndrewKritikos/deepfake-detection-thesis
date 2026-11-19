import timm
import torch
import torch.nn as nn
from timm.layers import resample_abs_pos_embed

class ChainHybrid(nn.Module):
    def __init__(
            self,
            cnn_name: str = "efficientnet_b0",
            vit_name: str = "deit_small_patch16_224",
            cnn_blocks: int = 5,
            vit_depth: int = 6,
            num_classes: int = 1,
            pretrained: bool = True,
            drop_rate: float = 0.1,
            img_size: int = 224
    ):
        super().__init__()

        #   CNN stem 
        m = timm.create_model(cnn_name, pretrained=pretrained)
        self.cnn = nn.Sequential(m.conv_stem, m.bn1, *m.blocks[:cnn_blocks])
        cnn_ch = m.blocks[cnn_blocks -1][-1].conv_pwl.out_channels
        stride = 16
        grid = img_size // stride
        del m 

        #Donor ViT I only borrow its CLS token, post-embed, blocks and final norm
        vit = timm.create_model(vit_name, pretrained=pretrained)
        dim = vit.embed_dim
        assert vit_depth <= len(vit.blocks), 'vit_depth larger than the donor ViT depth'

        #Bridge: CNN channels -> ViT embedding dim
        self.proj = nn.Conv2d(cnn_ch, dim, kernel_size=1)
        self.proj_norm = nn.LayerNorm(dim)

        #Transormer
        self.cls_token = nn.Parameter(vit.cls_token.detach().clone())
        pos = resample_abs_pos_embed(
            vit.pos_embed.detach().clone(), new_size=(grid, grid), num_prefix_tokens=1
        )
        self.pos_embed = nn.Parameter(pos)
        self.blocks = nn.Sequential(*vit.blocks[:vit_depth])
        self.norm = vit.norm

        #Classification head
        self.drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(dim, num_classes)

        del vit

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x) #[B, C, H, W]
        t = self.proj(f).flatten(2).transpose(1, 2) #[B, N, D]
        t = self.proj_norm(t)
        cls = self.cls_token.expand(t.shape[0], -1, -1)
        t = torch.cat([cls, t], dim=1) + self.pos_embed #[B, N+1, D]
        t = self.blocks(t)
        t = self.norm(t)
        return t[:, 0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(self.forward_features(x)))
    
    def param_groups(self, lr_backbone:float, lr_head:float, weight_decay: float = 0.05):
        """
        Differential learning rates: pretrained parts get a small LR, new layers a larger one
        """
        backbone = [
            *self.cnn.parameters(),
            *self.blocks.parameters(),
            *self.norm.parameters(),
            self.cls_token,
            self.pos_embed,
        ]
        new = [
            *self.proj.parameters(),
            *self.proj_norm.parameters(),
            *self.head.parameters(),
        ]
        return [
            {'params': backbone, 'lr': lr_backbone, 'weight_decay': weight_decay},
            {'params': new, 'lr': lr_head, 'weight_decay': weight_decay}
        ]
    
def count_params(model: nn.Module) -> float:
        '''Trainable params in millions
        '''
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    
if __name__ == "__main__":
    model = ChainHybrid(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print('output:', tuple(out.shape), '| params (M):', round(count_params(model), 2))