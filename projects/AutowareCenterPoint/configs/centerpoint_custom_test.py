custom_imports = dict(
    imports=[
        'projects.AutowareCenterPoint.centerpoint',
        'projects.AutowareCenterPoint.datasets',
        'projects.AutowareCenterPoint.evaluation'
    ],
    allow_failed_imports=False)

auto_scale_lr = dict(base_batch_size=32, enable=False)
backend_args = None

use_dim = [
    0,
    1,
    2,
    4,
]
use_dim_num = 5
load_dim_num = 5
point_cloud_range = [
    -51.2,
    -51.2,
    -5.0,
    51.2,
    51.2,
    3.0,
]

class_names = [
    'car',
    'truck',
    'bus',
    'bicycle',
    'pedestrian',
]

voxel_size = [
    0.2,
    0.2,
    8.0,
]

data_prefix = dict(img='', pts='', sweeps='')
data_root = 'data/sample_dataset/'
dataset_type = 'T4Dataset'

train_ann_file = dataset_type + '_infos_train.pkl'
val_ann_file = dataset_type + '_infos_val.pkl'
test_ann_file = dataset_type + '_infos_train.pkl'

db_info_path = data_root + '/' + dataset_type + '_dbinfos_train.pkl'

db_sampler = dict(
    backend_args=None,
    classes=class_names,
    data_root=data_root,
    info_path=db_info_path,
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=load_dim_num,
        type='LoadPointsFromFile',
        use_dim=use_dim),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(
            bicycle=5, bus=5, car=5, pedestrian=5, truck=5)),
    rate=1.0,
    sample_groups=dict(bicycle=6, bus=4, car=2, pedestrian=2, truck=3))
default_hooks = dict(
    checkpoint=dict(interval=1, save_optimizer=True, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
eval_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=load_dim_num,
        type='LoadPointsFromFile',
        use_dim=use_dim_num),
    dict(
        backend_args=None,
        sweeps_num=0,
        test_mode=True,
        type='LoadPointsFromMultiSweeps'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
input_modality = dict(use_camera=False, use_lidar=True)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
lr = 0.0001
metainfo = dict(classes=class_names)
model = dict(
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,
            max_voxels=(
                30000,
                40000,
            ),
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    pts_backbone=dict(
        conv_cfg=dict(bias=False, type='Conv2d'),
        in_channels=32,
        layer_nums=[
            3,
            5,
            5,
        ],
        layer_strides=[
            1,
            2,
            2,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            64,
            128,
            256,
        ],
        type='SECOND'),
    pts_bbox_head=dict(
        bbox_coder=dict(
            code_size=9,
            max_num=500,
            out_size_factor=1,
            pc_range=[
                -51.2,
                -51.2,
            ],
            post_center_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
                10.0,
            ],
            score_threshold=0.1,
            type='CenterPointBBoxCoder',
            voxel_size=[
                0.2,
                0.2,
            ]),
        common_heads=dict(
            dim=(
                3,
                2,
            ),
            height=(
                1,
                2,
            ),
            reg=(
                2,
                2,
            ),
            rot=(
                2,
                2,
            ),
            vel=(
                2,
                2,
            )),
        in_channels=384,
        loss_bbox=dict(
            loss_weight=0.25, reduction='mean', type='mmdet.L1Loss'),
        loss_cls=dict(reduction='mean', type='mmdet.GaussianFocalLoss'),
        norm_bbox=True,
        separate_head=dict(
            final_kernel=3, init_bias=-2.19, type='SeparateHead'),
        share_conv_channel=64,
        tasks=[
            dict(class_names=class_names, num_class=5),
        ],
        type='CenterHead'),
    pts_middle_encoder=dict(
        in_channels=32, output_shape=(
            512,
            512,
        ), type='PointPillarsScatter'),
    pts_neck=dict(
        in_channels=[
            64,
            128,
            256,
        ],
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN'),
        out_channels=[
            128,
            128,
            128,
        ],
        type='SECONDFPN',
        upsample_cfg=dict(bias=False, type='deconv'),
        upsample_strides=[
            1,
            2,
            4,
        ],
        use_conv_for_no_stride=True),
    pts_voxel_encoder=dict(
        feat_channels=[
            32,
            32,
        ],
        in_channels=4,
        legacy=False,
        norm_cfg=dict(eps=0.001, momentum=0.01, type='BN1d'),
        point_cloud_range=point_cloud_range,
        type='PillarFeatureNetAutoware',
        voxel_size=voxel_size,
        with_distance=False,
        use_voxel_center_z=False),
    test_cfg=dict(
        pts=dict(
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[
                4,
                12,
                10,
                1,
                0.85,
                0.175,
            ],
            nms_thr=0.2,
            nms_type='circle',
            out_size_factor=1,
            pc_range=[
                -51.2,
                -51.2,
            ],
            post_center_limit_range=[
                -61.2,
                -61.2,
                -10.0,
                61.2,
                61.2,
                10.0,
            ],
            post_max_size=83,
            pre_max_size=1000,
            score_threshold=0.1,
            voxel_size=[
                0.2,
                0.2,
            ])),
    train_cfg=dict(
        pts=dict(
            code_weights=[
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.2,
                0.2,
            ],
            dense_reg=1,
            gaussian_overlap=0.1,
            grid_size=[
                512,
                512,
                1,
            ],
            max_objs=500,
            min_radius=2,
            out_size_factor=1,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size)),
    type='CenterPoint')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01),
    type='OptimWrapper')
out_size_factor = 1
param_scheduler = [
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.001,
        type='CosineAnnealingLR'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=1e-08,
        type='CosineAnnealingLR'),
    dict(
        T_max=8,
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=8,
        eta_min=0.8947368421052632,
        type='CosineAnnealingMomentum'),
    dict(
        T_max=12,
        begin=8,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=1,
        type='CosineAnnealingMomentum'),
]

resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=test_ann_file,
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=[
            'car',
            'truck',
            'bus',
            'bicycle',
            'pedestrian',
        ]),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=load_dim_num,
                type='LoadPointsFromFile',
                use_dim=use_dim_num),
            dict(
                backend_args=None,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=0,
                type='LoadPointsFromMultiSweeps',
                use_dim=use_dim),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='T4Dataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=data_root + test_ann_file,
    backend_args=None,
    data_root=data_root,
    metric='bbox',
    type='NuScenesCustomMetric')
test_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=load_dim_num,
        type='LoadPointsFromFile',
        use_dim=use_dim_num),
    dict(
        backend_args=None,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=0,
        type='LoadPointsFromMultiSweeps',
        use_dim=use_dim),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=5)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            ann_file=train_ann_file,
            backend_args=None,
            box_type_3d='LiDAR',
            data_prefix=data_prefix,
            data_root=data_root,
            metainfo=dict(classes=class_names),
            pipeline=[
                dict(
                    backend_args=None,
                    coord_type='LIDAR',
                    load_dim=load_dim_num,
                    type='LoadPointsFromFile',
                    use_dim=use_dim_num),
                dict(
                    backend_args=None,
                    pad_empty_sweeps=True,
                    remove_close=True,
                    sweeps_num=0,
                    type='LoadPointsFromMultiSweeps',
                    use_dim=use_dim),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    db_sampler=dict(
                        backend_args=None,
                        classes=class_names,
                        data_root=data_root,
                        info_path=data_root + train_ann_file,
                        points_loader=dict(
                            backend_args=None,
                            coord_type='LIDAR',
                            load_dim=load_dim_num,
                            type='LoadPointsFromFile',
                            use_dim=use_dim),
                        prepare=dict(
                            filter_by_difficulty=[
                                -1,
                            ],
                            filter_by_min_points=dict(
                                bicycle=5, bus=5, car=5, pedestrian=5,
                                truck=5)),
                        rate=1.0,
                        sample_groups=dict(
                            bicycle=6, bus=4, car=2, pedestrian=2, truck=3)),
                    type='ObjectSample'),
                dict(
                    rot_range=[
                        -0.3925,
                        0.3925,
                    ],
                    scale_ratio_range=[
                        0.95,
                        1.05,
                    ],
                    translation_std=[
                        0,
                        0,
                        0,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    point_cloud_range=point_cloud_range,
                    type='PointsRangeFilter'),
                dict(
                    point_cloud_range=point_cloud_range,
                    type='ObjectRangeFilter'),
                dict(classes=class_names, type='ObjectNameFilter'),
                dict(type='PointShuffle'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            test_mode=False,
            type=dataset_type,
            use_valid_flag=True),
        type='CBGSDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=load_dim_num,
        type='LoadPointsFromFile',
        use_dim=use_dim_num),
    dict(
        backend_args=None,
        pad_empty_sweeps=True,
        remove_close=True,
        sweeps_num=0,
        type='LoadPointsFromMultiSweeps',
        use_dim=use_dim),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        db_sampler=dict(
            backend_args=None,
            classes=class_names,
            data_root=data_root,
            info_path=data_root + train_ann_file,
            points_loader=dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=load_dim_num,
                type='LoadPointsFromFile',
                use_dim=use_dim),
            prepare=dict(
                filter_by_difficulty=[
                    -1,
                ],
                filter_by_min_points=dict(
                    bicycle=5, bus=5, car=5, pedestrian=5, truck=5)),
            rate=1.0,
            sample_groups=dict(bicycle=6, bus=4, car=2, pedestrian=2,
                               truck=3)),
        type='ObjectSample'),
    dict(
        rot_range=[
            -0.3925,
            0.3925,
        ],
        scale_ratio_range=[
            0.95,
            1.05,
        ],
        translation_std=[
            0,
            0,
            0,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(point_cloud_range=point_cloud_range, type='PointsRangeFilter'),
    dict(point_cloud_range=point_cloud_range, type='ObjectRangeFilter'),
    dict(classes=class_names, type='ObjectNameFilter'),
    dict(type='PointShuffle'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=val_ann_file,
        backend_args=None,
        box_type_3d='LiDAR',
        data_prefix=data_prefix,
        data_root=data_root,
        metainfo=dict(classes=class_names),
        modality=dict(use_camera=False, use_lidar=True),
        pipeline=[
            dict(
                backend_args=None,
                coord_type='LIDAR',
                load_dim=load_dim_num,
                type='LoadPointsFromFile',
                use_dim=use_dim_num),
            dict(
                backend_args=None,
                pad_empty_sweeps=True,
                remove_close=True,
                sweeps_num=0,
                type='LoadPointsFromMultiSweeps',
                use_dim=use_dim),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='NuScenesDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=data_root + val_ann_file,
    backend_args=None,
    data_root=data_root,
    metric='bbox',
    type='NuScenesCustomMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])

work_dir = './work_dirs/centerpoint_custom_test'
