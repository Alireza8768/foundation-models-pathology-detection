# train_retinanet_h0_midogpp.py

"""
Training script for RetinaNet with H-Optimus-0 backbone on MIDOG++.

This script:
- loads an MMDetection config file
- initializes the MMDetection runner
- starts training

All model architecture, datasets, and training parameters
are defined in the config file.
"""


import os
import os.path as osp
import sys
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
from mmengine.logging import print_log


def main():
    
    project_root = Path(__file__).resolve().parents[2]
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    cfg_path = str(project_root / 'configs/retinanet_h0_midogpp.py')

    cfg = Config.fromfile(cfg_path)

    init_default_scope('mmdet')

    # Optional: Resume Training
    # cfg.resume = True

    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
