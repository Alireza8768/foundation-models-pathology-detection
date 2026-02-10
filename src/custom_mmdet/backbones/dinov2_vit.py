# custom_mmdet/backbones/dinov2_vit.py

import torch
import torch.nn as nn
import warnings
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from transformers import AutoModel, AutoImageProcessor

@MODELS.register_module()
class DINOv2Backbone(BaseModule):
    """
    Dino-V2-Giant backbone wrapper for MMDetection.

    Input:
        - Image tensor of shape (B, 3, H, W)
        - Example: (B, 3, 1008, 1008)

    Processing:
        - Loads pretrained Dino-V2-Giant via timm from Hugging Face Hub
        - Applies patch embedding (patch size = 14)
        - Removes extra tokens (e.g. CLS / register tokens)
        - Reshapes ViT tokens from (B, N, C) to a 2D feature map

    Output:
        - Single 2D feature map for detection neck
        - Shape: (B, C, H/14, W/14)
        - Example: (B, 1536, 72, 72)

    Note:
        - Output is returned as a tuple: (features,)
          to match MMDetection neck interface.
    """
    
    def __init__(
        self,
        model_id: str = "facebook/dinov2-giant",
        img_size: int = 1024,
        patch_size: int = 14,
        frozen: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        
        self.model_id = model_id
        self.img_size = img_size
        self.patch_size = patch_size
        self.frozen = frozen
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        print(f"Lade DINOv2-Modell: {model_id}")
        self.vit = AutoModel.from_pretrained(model_id)
        self.embed_dim = self.vit.config.hidden_size
        
        if frozen:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()
        
        print(f"DINOv2 Backbone initialisiert: embed_dim={self.embed_dim}")
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) in [0, 255]
        Returns:
            tuple[Tensor]: Multi-scale feature maps
        """
        # Normalisieren: [0, 255] -> ImageNet stats
        x = x.float() / 255.0
        x = (x - self.mean) / self.std
        
        # Forward durch ViT
        outputs = self.vit(pixel_values=x)
        last_hidden = outputs.last_hidden_state  # (B, N, C)
        
        B, N, C = last_hidden.shape
        
        # CLS-Token entfernen (erster Token)
        patch_tokens = last_hidden[:, 1:, :]  # (B, N-1, C)
        
        # Berechne räumliche Dimensionen
        H = self.img_size // self.patch_size
        W = self.img_size // self.patch_size
        
        # Prüfe ob Token-Anzahl korrekt ist
        expected_tokens = H * W
        actual_tokens = patch_tokens.shape[1]
        
        if actual_tokens != expected_tokens:
            raise ValueError(
                f"Falsche Token-Anzahl: erwartet {expected_tokens} (H={H}, W={W}), "
                f"bekommen {actual_tokens}. Stellen Sie sicher, dass die "
                f"Input-Größe {self.img_size}x{self.img_size} ist."
            )
        
        # Umformen zu räumlicher Feature-Map
        feature_map = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # Rückgabe als Tuple (MMDet erwartet das)
        return (feature_map,)

    def train(self, mode=True):
        """Override train to ensure frozen layers stay frozen"""
        super().train(mode)
        if self.frozen:
            # Stelle sicher, dass das ViT immer im eval mode ist
            self.vit.eval()
            for param in self.vit.parameters():
                param.requires_grad = False
        return self