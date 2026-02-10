# custom_mmdet/backbones/virchow_backbone.py

import math
from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
import timm
from timm.layers import SwiGLUPacked
from mmengine.model import BaseModule
from mmdet.registry import MODELS
import torch.nn.functional as F

@MODELS.register_module()
class VirchowBackbone(BaseModule):
    """
    Virchow backbone wrapper for MMDetection.

    Input:
        - Image tensor of shape (B, 3, H, W)
        - Example: (B, 3, 1008, 1008)

    Processing:
        - Loads pretrained Virchow via timm from Hugging Face Hub
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
        model_name: str = "hf-hub:paige-ai/Virchow",
        img_size: int = 1008,
        frozen: bool = True,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg)

        self.model_name = model_name
        self.img_size = img_size

        print(f"[VirchowBackbone] Lade Virchow-Modell über timm: {model_name}")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=nn.SiLU,
        )

        if hasattr(self.model, "patch_embed"):
            patch = getattr(self.model.patch_embed, "patch_size", None)
            if isinstance(patch, (tuple, list)):
                self.patch_size = patch[0]
            elif isinstance(patch, int):
                self.patch_size = patch
            else:
                self.patch_size = 14
        else:
            self.patch_size = 14

        if hasattr(self.model, "embed_dim"):
            self.embed_dim = self.model.embed_dim
        elif hasattr(self.model, "num_features"):
            self.embed_dim = self.model.num_features
        else:
            raise RuntimeError(
                "[VirchowBackbone] Konnte embed_dim nicht automatisch bestimmen."
            )

        print(
            f"[VirchowBackbone] img_size={self.img_size}, "
            f"patch_size={self.patch_size}, embed_dim={self.embed_dim}"
        )

        if frozen:
            for p in self.model.parameters():
                p.requires_grad = False
            print("[VirchowBackbone] Backbone-Parameter sind eingefroren (frozen=True).")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Forward-Pass.

        Args:
            x: Tensor [B, 3, H, W]

        Returns:
            Tuple mit einer Feature-Map [B, C, H_feat, W_feat]
        """
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            )

        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict):
                if "x" in feats:
                    feats = feats["x"]
                elif "last_hidden_state" in feats:
                    feats = feats["last_hidden_state"]
        else:
            feats = self.model(x)

        B, N, C = feats.shape
        num_patches = N
        has_cls = False

        root = int(math.sqrt(N))
        if root * root != N:
            num_patches = N - 1
            feats_patches = feats[:, 1:, :]
            has_cls = True
        else:
            feats_patches = feats

        H = int(math.sqrt(num_patches))
        W = H
        if H * W != num_patches:
            raise RuntimeError(
                f"[VirchowBackbone] Kann Patch-Grid nicht bestimmen: num_patches={num_patches}"
            )
        feats_patches = feats_patches.permute(0, 2, 1).reshape(B, C, H, W)
        
        return (feats_patches,)
