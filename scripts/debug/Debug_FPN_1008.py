# scripts/debug/Debug_FPN_1008.py

"""
Debug utility to inspect backbone and FPN feature map shapes, effective strides,
and anchor configuration for the 1008*1008 setup.
"""


import os
from pathlib import Path
import torch
from mmengine.config import Config
from mmdet.registry import MODELS
import sys

project_root = Path(__file__).resolve().parents[2]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
        
CFG_PATH = str(project_root / 'configs/faster_rcnn_h0_midogpp.py')

def main():
    cfg = Config.fromfile(CFG_PATH)
    model = MODELS.build(cfg.model)
    model.eval()

    img = torch.randn(1, 3, 1008, 1008)

    with torch.no_grad():
        feats = model.backbone(img)
        print("=== BACKBONE OUTPUTS ===")
        if isinstance(feats, dict):
            feats_list = list(feats.values())
        elif isinstance(feats, (list, tuple)):
            feats_list = list(feats)
        else:
            feats_list = [feats]

        for i, f in enumerate(feats_list):
            h, w = f.shape[-2:]
            IMG = 1008
            eff_stride = IMG // h

            print(f"Backbone[{i}]: shape={f.shape}, eff_stride≈{eff_stride}px")

        neck_feats = model.neck(feats_list)
        print("\n=== NECK / FPN OUTPUTS ===")
        featmap_sizes = []
        for i, f in enumerate(neck_feats):
            h, w = f.shape[-2:]
            IMG = 1008
            eff_stride = IMG // h
            featmap_sizes.append((h, w))
            print(f"P{i+3}: shape={f.shape}, eff_stride≈{eff_stride}px")

        ag = model.bbox_head.prior_generator
        print("\n=== ANCHOR-GENERATOR ===")
        print("Configured strides:", ag.strides)

        anchors = ag.grid_priors(
            featmap_sizes,
            dtype=torch.float32,
            device='cpu'
        )

        for lvl, a in enumerate(anchors):
            print(f"\nLevel P{lvl+3}:")
            print("  Anzahl Anchors:", a.shape[0])
            print("  Beispiel-Anker (erste 3):")
            print(a[:3])

if __name__ == "__main__":
    main()
