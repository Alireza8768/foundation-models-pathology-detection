# Data directory

This repository does **not** include the MIDOG++ images or generated patches.

Place your local data here, e.g.:

- `data/Datensatz/` – images / patches (not tracked)
- `data/coco_annotations/` – COCO-format JSON files (not tracked by default)

## Expected structure (example)

data/
├── Datensatz/                 # images or extracted patches
│   └── ...
└── coco_annotations/
    └── patches_1008/
        ├── midogpp_train.json
        ├── midogpp_val.json
        └── midogpp_test.json

## Notes

- The MMDetection configs assume `data_root = "./data/"`.
- If you use a different structure, update `data_root`, `ann_file`, and `data_prefix` in the configs.
