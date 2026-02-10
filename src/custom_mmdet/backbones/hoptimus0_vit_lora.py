# custom_mmdet/backbones/hoptimus0_vit_lora.py

import re
import torch
import torch.nn.functional as F
import timm
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from peft import LoraConfig, get_peft_model

@MODELS.register_module()
class H0Backbone(BaseModule):
    """
    H-Optimus-0 (LoRA fine-tuning) backbone wrapper for MMDetection.

    Input:
        - Image tensor of shape (B, 3, H, W)
        - Example: (B, 3, 1008, 1008)

    Processing:
        - Loads pretrained H-optimus-0 via timm from Hugging Face Hub
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
        model_name="hf-hub:bioptimus/H-optimus-0",
        patch_size=14,
        frozen=True,
        auto_pad=True,
        dynamic_img_size=True,

        # --- LoRA params ---
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        lora_target_modules=("qkv",),

        lora_last_n_blocks=12, 
        use_grad_checkpointing=False, 

        init_cfg=None,
    ):
        super().__init__(init_cfg=None)

        self.model_name = model_name
        self.patch_size = patch_size
        self.auto_pad = auto_pad

        print(f"[H0Backbone] Lade H0 über timm: {model_name}")
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="",
            dynamic_img_size=dynamic_img_size
        )

        self.embed_dim = getattr(self.backbone, "embed_dim", getattr(self.backbone, "num_features", None))
        if self.embed_dim is None:
            raise RuntimeError("[H0Backbone] Konnte embed_dim/num_features nicht bestimmen.")

        if use_lora:
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is None:
                raise RuntimeError("[H0Backbone] backbone hat kein .blocks; kann lora_last_n_blocks nicht anwenden.")

            n_blocks = len(blocks)
            if lora_last_n_blocks is None or lora_last_n_blocks <= 0 or lora_last_n_blocks >= n_blocks:
                start = 0
            else:
                start = n_blocks - lora_last_n_blocks

            target_patterns = []
            for i in range(start, n_blocks):
                for tm in lora_target_modules:
                    target_patterns.append(f"blocks.{i}.attn.{tm}")

            print(
                f"[H0Backbone] LoRA aktiv: r={lora_r}, alpha={lora_alpha}, "
                f"targets={list(lora_target_modules)}, last_n_blocks={lora_last_n_blocks} "
                f"(apply blocks {start}..{n_blocks-1})"
            )

            lora_cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=target_patterns,
            )

            if use_grad_checkpointing:
                if hasattr(self.backbone, "set_grad_checkpointing"):
                    self.backbone.set_grad_checkpointing(True)
                    print("[H0Backbone] Grad checkpointing ON")
                elif hasattr(self.backbone, "grad_checkpointing"):
                    self.backbone.grad_checkpointing = True
                    print("[H0Backbone] Grad checkpointing ON")

            self.backbone = get_peft_model(self.backbone, lora_cfg)
            self.backbone.print_trainable_parameters()

        if frozen:
            for name, p in self.backbone.named_parameters():
                p.requires_grad = ("lora_" in name)

            if use_lora:
                self.backbone.eval()
            else:
                self.backbone.eval()

            print("[H0Backbone] Base eingefroren, LoRA trainierbar (falls aktiv).")

        trainable_lora = [
            n for n, p in self.backbone.named_parameters()
            if p.requires_grad
        ]

        print("[H0Backbone] Trainable params sample:")
        for n in trainable_lora[:10]:
            print("   ", n)

        print(f"[H0Backbone] Trainable param count: {len(trainable_lora)}")

        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(f"[H0Backbone] trainable/total = {trainable}/{total}")

    def init_weights(self):
        pass

    def _maybe_pad(self, x: torch.Tensor) -> torch.Tensor:
        if not self.auto_pad:
            return x
        _, _, h, w = x.shape
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    def forward(self, x: torch.Tensor):
        x = self._maybe_pad(x)
        B, _, H, W = x.shape

        if (H % self.patch_size) != 0 or (W % self.patch_size) != 0:
            raise ValueError(f"[H0Backbone] Input {H}x{W} nicht durch patch_size={self.patch_size} teilbar.")

        feats = self.backbone.forward_features(x)

        if isinstance(feats, (tuple, list)):
            feats = feats[-1]
        if isinstance(feats, dict):
            feats = feats.get("x", list(feats.values())[-1])

        if not torch.is_tensor(feats):
            raise TypeError(f"[H0Backbone] forward_features returned {type(feats)} statt Tensor.")

        if hasattr(self.backbone, "norm") and callable(getattr(self.backbone, "norm")):
            feats = self.backbone.norm(feats)

        if feats.dim() == 3:
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size
            expected = grid_h * grid_w

            n_tokens = feats.shape[1]
            extra = n_tokens - expected

            if extra > 0:
                feats = feats[:, extra:, :]
            elif extra < 0:
                raise ValueError(
                    f"[H0Backbone] Zu wenige Tokens: tokens={n_tokens} expected={expected} "
                    f"(H={H},W={W},patch={self.patch_size})."
                )

            feats = feats.transpose(1, 2).contiguous().view(B, self.embed_dim, grid_h, grid_w)

        return (feats,)
