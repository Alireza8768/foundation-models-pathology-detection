# configs/retinanet_dinov2_midogpp.py

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.dinov2_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
    ],
    allow_failed_imports=False
)

_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

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
    data_preprocessor=dict(
        pad_size_divisor=1, 
    ),
    backbone=dict(
        _delete_=True,
        type='DINOv2Backbone',
        model_id='facebook/dinov2-giant',
        img_size=1008,
        patch_size=14,
        frozen=True,
    ),
    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125),
        norm='LN',
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[7, 14, 28, 56, 112],
            center_offset=0.5,
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=3000,
        min_bbox_size=0,
        score_thr=0.15,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=False, backend='pillow'),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction=['horizontal', 'vertical']
    ),
    dict(type='PackDetInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]

test_pipeline = val_pipeline


data_root = './data/'

train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='coco_annotations/patches_1008/midogpp_train.json',
        data_prefix=dict(img='Datensatz/'), 
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False),
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
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        betas=(0.9, 0.999),
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        T_max=40,
        eta_min=1e-7,
        begin=0,
        end=40,
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=1
)


val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco_annotations/patches_1008/midogpp_test.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    logger=dict(
        type='LoggerHook',
        interval=10,
        log_metric_by_epoch=True,
    ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(seed=42, deterministic=False, diff_rank_seed=True)

resume = False

work_dir = './outputs/work_dirs/retinanet_dinov2_1008_40epochs'