# custom_mmdet/backbones/uni_vit.py

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
import timm
from mmengine.model import BaseModule
from mmdet.registry import MODELS

@MODELS.register_module()
class UNIBackbone(BaseModule):
    """
    UNI backbone wrapper for MMDetection.

    Input:
        - Image tensor of shape (B, 3, H, W)
        - Example: (B, 3, 1008, 1008)

    Processing:
        - Loads pretrained UNI via timm from Hugging Face Hub
        - Applies patch embedding (patch size = 14)
        - Removes extra tokens (e.g. CLS / register tokens)
        - Reshapes ViT tokens from (B, N, C) to a 2D feature map

    Outpu:
        - Single 2D feature map for detection neck
        - Shape: (B, C, H/14, W/14)
        - Example: (B, 1536, 72, 72)

    Note:
        - Output is returned as a tuple: (features,)
          to match MMDetection neck interface.
    """

    def __init__(
        self,
        model_name: str = "hf-hub:MahmoodLab/uni",
        img_size: int = 1024,
        frozen: bool = True,
        init_values: float = 1e-5,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg)

        self.model_name = model_name
        self.img_size = img_size

        print(f"[UNIBackbone] Lade UNI über timm: {model_name}")
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            init_values=init_values,
            dynamic_img_size=True,
            num_classes=0,
        )

        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_size"):
            ps = self.model.patch_embed.patch_size
            self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        else:
            self.patch_size = 16

        self.embed_dim = getattr(self.model, "embed_dim", getattr(self.model, "num_features", None))
        
        print(f"[UNIBackbone] Architektur bereit: img_size={self.img_size}, "
              f"patch_size={self.patch_size}, embed_dim={self.embed_dim}")

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            print("[UNIBackbone] Parameter erfolgreich eingefroren.")

        self.model.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), 
                            mode="bilinear", align_corners=False)

        feats = self.model.forward_features(x)
        
        if hasattr(self.model, "norm"):
            feats = self.model.norm(feats)

        B, N, C = feats.shape
        H = W = self.img_size // self.patch_size
        expected_patches = H * W

        if N == expected_patches + 1:
            feats = feats[:, 1:, :]
        elif N != expected_patches:
            feats = feats[:, -expected_patches:, :]

        # 5. [B, N, C] -> [B, C, H, W] für den Neck (SimpleFeaturePyramid)
        feats = feats.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        return (feats,)