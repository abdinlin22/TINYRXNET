norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ResNetWithSAMRXLAST',
        depth=101,
        sam_checkpoint='sam_vit_b.pth',
        freeze_sam=True,
        fuse_mode='concat', 
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=2305,  
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=512,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='TverskyLoss', loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1281,  
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='TverskyLoss', loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

dataset_type = 'TinyObjectSegmentationDataset'
data_root = './'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

resize_size = (1024, 2048)
crop_size = (512, 1024)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=resize_size),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=resize_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/train/Images',
        ann_dir='dataset/train/Labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/test/Images',
        ann_dir='dataset/test/Labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='dataset/test/Images',
        ann_dir='dataset/test/Labels',
        pipeline=test_pipeline)
)

log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

optimizer = dict(type='Adam', lr=1e-4, weight_decay=1e-4)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=3000)
checkpoint_config = dict(by_epoch=False, interval=1500)

evaluation = dict(interval=50, metric=['mIoU', 'mDice'], pre_eval=True, save_best='mIoU')

seed = 0
gpu_ids = range(1)
device = None
