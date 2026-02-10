auto_scale_lr = dict(base_batch_size=16)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'src.custom_mmdet.backbones.hoptimus0_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
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
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 12
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
    as_two_stage=True,
    backbone=dict(frozen=True, type='H0Backbone'),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=1,
        sync_cls_avg_factor=True,
        type='DINOHead'),
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
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_layers=6),
    neck=dict(
        in_channels=1536,
        norm='LN',
        out_channels=256,
        scale_factors=(
            2.0,
            1.0,
            0.5,
            0.25,
        ),
        type='SimpleFeaturePyramid'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=50),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
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
        data_root='./data/',
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
            dict(keep_ratio=False, scale=(
                1008,
                1008,
            ), type='Resize'),
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
    data_root='./data/',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        1008,
        1008,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet', max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='coco_annotations/patches_1008/midogpp_train.json',
        backend_args=None,
        data_prefix=dict(img='Datensatz/'),
        data_root='./data/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='coco_annotations/patches_1008/midogpp_val.json',
        backend_args=None,
        data_prefix=dict(img='Datensatz/'),
        data_root='./data/',
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
            dict(keep_ratio=False, scale=(
                1008,
                1008,
            ), type='Resize'),
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
    data_root='./data/',
    format_only=False,
    metric='bbox',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        1008,
        1008,
    ), type='Resize'),
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
work_dir = './outputs/work_dirs/dino_h0_1008_40epochs'
