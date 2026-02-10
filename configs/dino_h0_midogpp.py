# configs/dino_h0_midogpp.py

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.hoptimus0_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::dino/dino-4scale_r50_8xb2-12e_coco.py'

img_scale = (1008, 1008)

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

data_preprocessor = dict(
    type='DetDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
)

model = dict(
    data_preprocessor=dict(pad_size_divisor=1),

    backbone=dict(
        _delete_=True,
        type='H0Backbone',
        frozen=True,
    ),

    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25),  # DINO typischerweise 4 levels
        norm='LN',
    ),
    
    test_cfg=dict(max_per_img=50),
    
    bbox_head=dict(
        num_classes=1,

    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='PackDetInputs'),
]

test_pipeline = val_pipeline

data_root = './data/'

train_dataloader = dict(
    batch_size=1, 
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_train.json',
        data_prefix=dict(img='Datensatz/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_val.json',
        data_prefix=dict(img='Datensatz/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_test.json',
        data_prefix=dict(img='Datensatz/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    )
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=40, eta_min=1e-7, begin=0, end=40),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_val.json',
    metric='bbox',
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_test.json',
    metric='bbox',
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
)

randomness = dict(seed=42, deterministic=False, diff_rank_seed=True)

resume = False
work_dir = './outputs/work_dirs/dino_h0_1008_40epochs'
