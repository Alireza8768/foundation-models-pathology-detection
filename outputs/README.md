# Outputs directory

This directory contains all outputs generated during training and evaluation.
It is intentionally excluded from version control.

## Contents

- `work_dirs/`  
  Automatically created by MMDetection.
  Stores training-related artifacts such as checkpoints, logs, and evaluation results.

- `test/`  
  Output directory created by evaluation scripts.
  Used to store prediction files, plots, and result summaries.

## Configuration

### Training outputs

Training outputs are controlled via the `work_dir` parameter in each MMDetection
configuration file, for example:

```python
work_dir = "./outputs/work_dirs/faster_rcnn_h0_1008_40epochs"
