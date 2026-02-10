# Scripts

This directory contains helper scripts used during development, training, and evaluation.

## Structure

- `train/`  
  Training entry points. Each script typically launches a specific experiment/config.

- `eval/`  
  Evaluation and analysis scripts (e.g., official COCO evaluation, threshold sweeps, PR/F1 plots).

- `debug/`  
  Small utilities for inspecting intermediate feature maps and verifying
  architectural assumptions (e.g. FPN strides).

## Usage

Run scripts from the project root (`Training/`) to ensure relative paths work as expected:

```bash
cd Training
python scripts/train/train_faster_rcnn_h0_midogpp.py
