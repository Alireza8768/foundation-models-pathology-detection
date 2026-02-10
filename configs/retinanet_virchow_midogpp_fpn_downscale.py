# configs/retinanet_virchow_midogpp_fpn_downscale.py

auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'src.custom_mmdet.backbones.virchow_backbone',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_size_divisor=1,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='DetDataPreProcessor')
data_root = './data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        interval=1,
        max_keep_ckpts=3,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(
        _scope_='mmdet',
        interval=10,
        log_metric_by_epoch=True,
        type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scale = (
    1008,
    1008,
)
img_scales = [
    (
        1333,
        800,
    ),
    (
        666,
        400,
    ),
    (
        2000,
        1200,
    ),
]
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=('mitotic figure', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    _scope_='mmdet',
    backbone=dict(
        frozen=True,
        img_size=1008,
        model_name='hf-hub:paige-ai/Virchow',
        type='VirchowBackbone'),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                14,
                28,
                56,
                112,
                201,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        num_classes=1,
        stacked_convs=4,
        type='RetinaHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            1280,
        ],
        num_outs=5,
        out_channels=256,
        start_level=0,
        type='FPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.5, type='nms'),
        nms_pre=1000,
        score_thr=0.15),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.4,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='RetinaNet')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        T_max=40,
        begin=0,
        by_epoch=True,
        end=40,
        eta_min=1e-07,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=True, seed=42)
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='coco_annotations/patches_1008/midogpp_test.json',
        backend_args=None,
        data_prefix=dict(img='Datensatz/'),
        data_root='./',
        metainfo=dict(
            classes=('mitotic figure', ), palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1008,
                1008,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco_annotations/patches_1008/midogpp_test.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1008,
        1008,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet', max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=6,
    dataset=dict(
        _scope_='mmdet',
        ann_file='coco_annotations/patches_1008/midogpp_train.json',
        backend_args=None,
        data_prefix=dict(img='Datensatz/'),
        data_root='./',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(
            classes=('mitotic figure', ), palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                backend='pillow',
                keep_ratio=False,
                scale=(
                    1008,
                    1008,
                ),
                type='Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                ],
                prob=0.5,
                type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        backend='pillow',
        keep_ratio=False,
        scale=(
            1008,
            1008,
        ),
        type='Resize'),
    dict(direction=[
        'horizontal',
        'vertical',
    ], prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    _scope_='mmdet',
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.5, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmdet',
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    1333,
                    800,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    666,
                    400,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    2000,
                    1200,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='coco_annotations/patches_1008/midogpp_val.json',
        backend_args=None,
        data_prefix=dict(img='Datensatz/'),
        data_root='./',
        metainfo=dict(
            classes=('mitotic figure', ), palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                1008,
                1008,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='data/coco_annotations/patches_1008/midogpp_val.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        1008,
        1008,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs'),
]
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './outputs/work_dirs/retinanet_virchow_1008_mitosis_40Epochs'
