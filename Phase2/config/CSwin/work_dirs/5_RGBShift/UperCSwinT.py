augs_num = 5
augs_epoch = 20
augs = [
    dict(type='CLAHE', p=1.0),
    dict(type='RandomGamma', p=1.0),
    dict(type='HueSaturationValue', p=1.0),
    dict(type='ChannelDropout', p=1.0),
    dict(type='ChannelShuffle', p=1.0),
    dict(type='RGBShift', p=1.0),
    dict(type='ShiftScaleRotate', p=1.0),
    dict(type='RandomRotate90', p=1.0),
    dict(type='PiecewiseAffine', p=1.0),
    dict(type='CoarseDropout', max_height=8, max_width=8, p=1.0),
    dict(type='ElasticTransform', border_mode=0, p=1.0),
    dict(type='ElasticTransform', p=1.0),
    dict(type='GridDistortion', border_mode=0, p=1.0),
    dict(type='RandomCrop', height=300, width=300, p=1.0),
    dict(type='OpticalDistortion', distort_limit=0.5, p=1.0)
]
alb_transform = [
    dict(type='VerticalFlip', p=0.3),
    dict(type='HorizontalFlip', p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='GaussNoise', p=1.0),
            dict(type='GaussianBlur', p=1.0),
            dict(type='Blur', p=1.0)
        ],
        p=0.3),
    dict(type='OneOf', transforms=[dict(type='RGBShift', p=1.0)], p=0.3)
]
dataset_type = 'CustomDataset'
data_root = '/opt/ml/segmentation/input/mmseg/'
classes = [
    'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
    'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
]
palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0],
           [64, 0, 128], [64, 0, 192], [192, 128, 64], [192, 192, 128],
           [64, 64, 128], [128, 0, 192]]
img_norm_cfg = dict(
    mean=[117.551, 112.259, 106.825],
    std=[59.866, 58.944, 62.162],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512)),
    dict(
        type='Albu',
        transforms=[
            dict(type='VerticalFlip', p=0.3),
            dict(type='HorizontalFlip', p=0.3),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='GaussNoise', p=1.0),
                    dict(type='GaussianBlur', p=1.0),
                    dict(type='Blur', p=1.0)
                ],
                p=0.3),
            dict(
                type='OneOf', transforms=[dict(type='RGBShift', p=1.0)], p=0.3)
        ]),
    dict(type='RandomFlip', prob=0.3),
    dict(
        type='Normalize',
        mean=[117.551, 112.259, 106.825],
        std=[59.866, 58.944, 62.162],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[117.551, 112.259, 106.825],
                std=[59.866, 58.944, 62.162],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
            dict(
                type='Normalize',
                mean=[117.551, 112.259, 106.825],
                std=[59.866, 58.944, 62.162],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/segmentation/input/mmseg/images/training',
        ann_dir='/opt/ml/segmentation/input/mmseg/annotations/training',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(512, 512)),
            dict(
                type='Albu',
                transforms=[
                    dict(type='VerticalFlip', p=0.3),
                    dict(type='HorizontalFlip', p=0.3),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='GaussNoise', p=1.0),
                            dict(type='GaussianBlur', p=1.0),
                            dict(type='Blur', p=1.0)
                        ],
                        p=0.3),
                    dict(
                        type='OneOf',
                        transforms=[dict(type='RGBShift', p=1.0)],
                        p=0.3)
                ]),
            dict(type='RandomFlip', prob=0.3),
            dict(
                type='Normalize',
                mean=[117.551, 112.259, 106.825],
                std=[59.866, 58.944, 62.162],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/segmentation/input/mmseg/images/validation',
        ann_dir='/opt/ml/segmentation/input/mmseg/annotations/validation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[117.551, 112.259, 106.825],
                        std=[59.866, 58.944, 62.162],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        classes=[
            'Background', 'General trash', 'Paper', 'Paper pack', 'Metal',
            'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery',
            'Clothing'
        ],
        palette=[[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64],
                 [128, 0, 0], [64, 0, 128], [64, 0, 192], [192, 128, 64],
                 [192, 192, 128], [64, 64, 128], [128, 0, 192]],
        type='CustomDataset',
        reduce_zero_label=False,
        img_dir='/opt/ml/segmentation/input/mmseg/test',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[117.551, 112.259, 106.825],
                        std=[59.866, 58.944, 62.162],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
lr = 0.0001
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=300,
    warmup_ratio=0.1,
    min_lr_ratio=7e-06)
total_epochs = 20
expr_name = '5_RGBShift'
dist_params = dict(backend='nccl')
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=20)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='segm_augs', name='5_RGBShift', entity='ark10806'))
    ])
custom_hooks = [dict(type='NumClassCheckHook')]
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
evaluation = dict(metric='mIoU', pre_eval=True, save_best='mIoU')
work_dir = './work_dirs/5_RGBShift'
gpu_ids = range(0, 1)
emb = 64
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=
    'https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth',
    backbone=dict(
        type='CSWin',
        embed_dim=64,
        patch_size=4,
        depth=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
