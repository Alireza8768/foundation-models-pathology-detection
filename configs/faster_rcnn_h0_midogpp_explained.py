# faster_rcnn_h0_midogpp_explained.py

custom_imports = dict(
    imports=[
        'src.custom_mmdet.backbones.hoptimus0_vit',
        'src.custom_mmdet.necks.simple_feature_pyramid',
    ],
    allow_failed_imports=False
)

 
# Base config: Faster R-CNN template (loads default heads, train/test settings)
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

img_scale = (1008, 1008)

""" 
"categories": [
  { "id": 1, "name": "mitotic figure" }
]

palette ist eine RGB-Farbe:
    Rot = 220
    Grün = 20
    Blau = 60

Wenn man mehrere Kategorien hat:
    classes=('a','b','c')
    palette=[(255,0,0),(0,255,0),(0,0,255)] 

"""

metainfo = dict(
    classes=('mitotic figure',),
    palette=[(220, 20, 60)],
)

""" pad_size_divisor = 1 Heißt:

    kein automatisches Padding
    Bild bleibt exakt 1008 x 1008
"""

data_preprocessor = dict(
    type='DetDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
)

model = dict(
    data_preprocessor=dict(pad_size_divisor=1),

    #_delete_=True bedeutet:
    #Lösche den kompletten Backbone aus der Base-Config und ersetze ihn vollständig. 

    backbone=dict(
        _delete_=True,
        type='H0Backbone',
        frozen=True,
    ),

    # in_channels=1536 - was bedeutet das?
    #     ist die Kanalzahl der Backbone-Features (Feature-Dimension).
        
    # out_channels=256 - warum 256?
    #     Detektoren wie Faster R-CNN erwarten (typisch):
    #     FPN-Level alle auf gleicher Kanalzahl (meist 256)
        
    # scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125)
    #     2.0 → 144*144 (Upsample, höher aufgelöst)
    #     1.0 → 72*72 (original)
    #     0.5 → 36*36
    #     0.25→ 18*18
    #     0.125→ 9*9
    #     Ergebnis: 5 Feature-Levels (wie eine FPN-Pyramide)

    # norm='LN' - warum LayerNorm?
    #     Bei CNN-FPN nimmt man oft BatchNorm/GroupNorm. 

    neck=dict(
        _delete_=True,
        type='SimpleFeaturePyramid',
        in_channels=1536,
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25, 0.125),
        norm='LN',
    ),

    # RPN + ROI Head (Faster R-CNN)
    
    # Sie schaut auf die Feature Maps Legt dort viele Anker-Boxen
    # Sagt: Objekt / kein Objekt
    # Grobe Box-Position
    
    # strides=[7, 14, 28, 56, 112] Das muss exakt zu deinem Neck passen.
    #     Feature Maps:
    #     144*144 → stride 7
    #     72*72 → stride 14
    #     36*36 → stride 28
    #     18*18 → stride 56
    #     9*9 → stride 112
    #     Stride = wie viele Pixel im Originalbild ein Schritt im Feature entspricht
    
    # scales=[4, 8, 16]
    #     Das sind Basisgrößen der Anker, relativ zum Stride.
    #     Beispiel bei stride=14:
    #     14 * 4 = 56 px
    #     14 * 8 = 112 px
    #     14 * 16 = 224 px
    #     Gut für kleine bis mittlere Objekte
    #     ratios=[0.5, 1.0, 2.0]
    #         Form der Box:
    #         0.5 → breit
    #         1.0 → quadratisch
    #         2.0 → hoch
    #         ungefähr rund → ratio 1.0 ist besonders wichtig 

    rpn_head=dict(
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            strides=[7, 14, 28, 56, 112],
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
    ),
    
    # roi_head – zweite Stufe (klassifizieren & verfeinern)
    
    # bbox_roi_extractor=:
    #     bekommt Vorschläge aus der RPN
    #     weiß, welche Feature Map benutzt werden soll
    #     schneidet die richtigen Regionen aus
    #     ➡️ featmap_stride Muss identisch zu RPN/Neck sein
        
    # num_classes=1
    #     eine Klasse: mitotic figure 

    roi_head = dict(
        bbox_roi_extractor=dict(
            featmap_strides=[7, 14, 28, 56, 112],
        ),
        bbox_head=dict(
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
        )
    ),

    # train_cfg beantwortet nur Trainingsfragen, z. B.:
    #     Welche Anchors gelten als positiv / negativ?
    #     Wie viele Beispiele werden pro Batch gelernt?
    #     Welche Boxen dürfen weitergegeben werden?
    #     Wie streng sind die Regeln?
        
    # train_cfg
    # ├── rpn        → Wie lernt die RPN?
    # ├── rpn_proposal → Welche Vorschläge gehen weiter?
    # └── rcnn       → Wie lernt der ROI Head? 

    train_cfg=dict(
            

        #    IoU = Überlappung zwischen:
        #         Anchor
        #         Ground-Truth-Box
        
        #     | IoU        | Bedeutung                       |
        #     | ---------- | ------------------------------- |
        #     | ≥ 0.7      | **positiv** (lernt Objekt)      |
        #     | ≤ 0.3      | **negativ** (lernt Hintergrund) |
        #     | dazwischen | ignoriert                       |

        #     match_low_quality=True:
        #         verhindert, dass kleine Objekte komplett ignoriert werden 

        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            
            # RPN erzeugt sehr viele Anchors (tausende).
            #     pro Bild 256 Anchors
            #     davon 50 % positiv, 50 % negativ 
            
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
         
        # Behalte Top-2000 Boxen (nach Score)
        # NMS entfernt überlappende Boxen
        # Maximal 1000 Boxen pro Bild 
       
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),

    # test_cfg
    #  ├── rpn   → Vorschläge filtern
    #  └── rcnn  → Endergebnis filtern 
     
    # RPN erzeugt sehr viele Vorschläge
    # Du behältst nur die 2000 besten (nms_pre)
    # NMS entfernt doppelte/überlappende Boxen
    # Maximal 1000 Boxen gehen weiter
   
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)


# Diese Pipelines beschreiben WAS mit einem Bild passiert,
# nachdem es gefunden wurde
# train_pipeline = [
#     LoadImageFromFile (lädt das Bild)
#     LoadAnnotations (lädt Bounding Boxes, with_bbox=True → nur Boxen, keine Masken)
#     Resize (bringt alle Bilder auf 1008×1008)
#     RandomFlip (Datenaugmentation)
#     PackDetInputs (packt alles in das Format)
# ]


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

    # filter_cfg=dict(filter_empty_gt=False) Was heißt das?
    #     Bilder ohne Mitosen bleiben im Training


train_dataloader = dict(
    batch_size=6,   
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
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


# Wrapper = “Container” um den Optimizer mit Zusatzfunktionen.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=35, norm_type=2),
)


# param_scheduler – wie verändert sich die LR über die Zeit?
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', by_epoch=True, T_max=40, eta_min=1e-7, begin=0, end=40),
]


# param_scheduler – wie verändert sich die LR über die Zeit?
# val_interval=1
#     nach jeder Epoche validieren

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)


# metric='bbox'
#     wir bewerten Bounding Boxes (Objekterkennung)
#     nicht Segmentation (segm)
#     nicht Keypoints (keypoints)

# format_only=False
#     bedeutet: wirklich evaluieren
#     es True wäre:
#         dann würden nur Ergebnisse exportiert (z. B. JSON)
#         aber keine mAP berechnet

# backend_args=None
#     kein spezielles Backend für Filesystem/Remote Storage 

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


# CheckpointHook
#     interval=1 → speichert jede Epoche ein Checkpoint
#     max_keep_ckpts=3 → behält nur die letzten 3 (spart Speicher)

# save_best='coco/bbox_mAP'
#     zusätzlich speichert er das beste Modell nach der Metrik:

# LoggerHook
#     interval=10 → loggt alle 10 Iterationen (nicht nur pro Epoche)
#     log_metric_by_epoch=True → Metriken pro Epoche sauber zusammengefasst 

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP', rule='greater'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

randomness = dict(seed=42, deterministic=False, diff_rank_seed=True)


# resume = False
#     Training startet immer neu
#     kein automatisches Fortsetzen 

resume = False

# Das ist dein Output-Ordner für:
#     Checkpoints
#     Logs
#     Tensorboard/JSON logs (je nach Hook)
#     Visualizations (wenn aktiviert) 

work_dir = './outputs/work_dirs/faster_rcnn_h0_1008_40epochs'
