import os
from os import path as osp

import numpy as np

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class T4Dataset(NuScenesDataset):
    METAINFO = {
        'classes': ('car', 'truck', 'bus', 'bicycle', 'pedestrian'),
        'version':
        'v1.0-trainval',
        'palette': [
            (255, 158, 0),  # Orange
            (255, 99, 71),  # Tomato
            (255, 140, 0),  # Darkorange
            (255, 127, 80),  # Coral
            (233, 150, 70),  # Darksalmon
        ]
    }

    def __init__(
        self,
        box_type_3d: str = 'LiDAR',
        load_type: str = 'frame_based',
        with_velocity: bool = True,
        use_valid_flag: bool = False,
        **kwargs,
    ) -> None:

        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity

        # TODO: Redesign multi-view data process in the future
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')
        self.load_type = load_type

        assert box_type_3d.lower() in ('lidar', 'camera')
        super().__init__(**kwargs)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process the `instances` field to
        `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.load_type == 'mv_image_based':
            info = super().parse_data_info(info)
        else:
            if self.modality['use_lidar']:
                info['lidar_points']['lidar_path'] = \
                    osp.join(
                        self.data_prefix.get('pts', ''),
                        info['lidar_points']['lidar_path'])

                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
                info['lidar_path'] = info['lidar_points']['lidar_path']
                if 'lidar_sweeps' in info:
                    for sweep in info['lidar_sweeps']:
                        file_suffix_splitted = sweep['lidar_points'][
                            'lidar_path'].split(os.sep)
                        file_suffix = os.sep.join(file_suffix_splitted[-4:])
                        if 'samples' in sweep['lidar_points']['lidar_path']:
                            sweep['lidar_points']['lidar_path'] = osp.join(
                                self.data_prefix['pts'], file_suffix)
                        else:
                            sweep['lidar_points']['lidar_path'] = info[
                                'lidar_points']['lidar_path']

            if self.modality['use_camera']:
                for cam_id, img_info in info['images'].items():
                    if 'img_path' in img_info:
                        if cam_id in self.data_prefix:
                            cam_prefix = self.data_prefix[cam_id]
                        else:
                            cam_prefix = self.data_prefix.get('img', '')
                        img_info['img_path'] = osp.join(
                            cam_prefix, img_info['img_path'])
                if self.default_cam_key is not None:
                    info['img_path'] = info['images'][
                        self.default_cam_key]['img_path']
                    if 'lidar2cam' in info['images'][self.default_cam_key]:
                        info['lidar2cam'] = np.array(
                            info['images'][self.default_cam_key]['lidar2cam'])
                    if 'cam2img' in info['images'][self.default_cam_key]:
                        info['cam2img'] = np.array(
                            info['images'][self.default_cam_key]['cam2img'])
                    if 'lidar2img' in info['images'][self.default_cam_key]:
                        info['lidar2img'] = np.array(
                            info['images'][self.default_cam_key]['lidar2img'])
                    else:
                        info['lidar2img'] = info['cam2img'] @ info['lidar2cam']

            if not self.test_mode:
                # used in training
                info['ann_info'] = self.parse_ann_info(info)
            if self.test_mode and self.load_eval_anns:
                info['eval_ann_info'] = self.parse_ann_info(info)

        return info
