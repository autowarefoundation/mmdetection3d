import argparse
import os
from typing import Dict, List, Optional, Tuple

import torch
from mmengine import Config
from mmengine.registry import MODELS, Registry
from mmengine.runner import Runner

from mmdet3d.apis import init_model
from mmdet3d.models.dense_heads.centerpoint_head import (CenterHead,
                                                         SeparateHead)
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator
from projects.AutowareCenterPoint.centerpoint.pillar_encoder_autoware import \
    PillarFeatureNetAutoware


def parse_args():
    parser = argparse.ArgumentParser(
        'Create autoware compatible onnx file from torch checkpoint ')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--ckpt', help='checkpoint weeight')
    parser.add_argument('--work-dir', help='the dir to save onnx files')

    args = parser.parse_args()
    return args


class CenterPointToONNX(object):

    def __init__(
        self,
        config: Config,
        checkpoint_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ):
        assert isinstance(
            config, Config), f'expected `mmcv.Config`, but got {type(config)}'
        _, ext = os.path.splitext(checkpoint_path)
        assert ext == '.pth', f'expected .pth model, but got {ext}'

        self.config = config
        self.checkpoint_path = checkpoint_path

        os.makedirs(output_path, exist_ok=True)
        self.output_path = output_path

    def save_onnx(self) -> None:
        # Overwrite models with Autoware's TensorRT compatible versions
        self.config.model.pts_voxel_encoder.type = 'PillarFeatureNetONNX'
        self.config.model.pts_bbox_head.type = 'CenterHeadONNX'
        self.config.model.pts_bbox_head.separate_head.type = 'SeparateHeadONNX'

        model = init_model(self.config, self.checkpoint_path, device='cuda:0')
        dataloader = Runner.build_dataloader(self.config.test_dataloader)
        batch_dict = next(iter(dataloader))

        voxel_dict = model.data_preprocessor.voxelize(
            batch_dict['inputs']['points'], batch_dict)
        input_features = model.pts_voxel_encoder.get_input_features(
            voxel_dict['voxels'], voxel_dict['num_points'],
            voxel_dict['coors']).to('cuda:0')

        # CenterPoint's PointPillar voxel encoder ONNX conversion
        pth_onnx_pve = os.path.join(
            self.output_path, 'pts_voxel_encoder_centerpoint_custom.onnx')
        torch.onnx.export(
            model.pts_voxel_encoder,
            (input_features, ),
            f=pth_onnx_pve,
            input_names=('input_features', ),
            output_names=('pillar_features', ),
            dynamic_axes={
                'input_features': {
                    0: 'num_voxels',
                    1: 'num_max_points'
                },
                'pillar_features': {
                    0: 'num_voxels'
                },
            },
            verbose=False,
            opset_version=11,
        )
        print(f'Saved pts_voxel_encoder onnx model: {pth_onnx_pve}')

        voxel_features = model.pts_voxel_encoder(input_features)
        voxel_features = voxel_features.squeeze()

        batch_size = voxel_dict['coors'][-1, 0] + 1
        x = model.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                     batch_size)

        # CenterPoint backbone's to neck ONNX conversion
        pts_backbone_neck_head = CenterPointHeadONNX(
            model.pts_backbone,
            model.pts_neck,
            model.pts_bbox_head,
        )

        pth_onnx_backbone_neck_head = os.path.join(
            self.output_path, 'pts_backbone_neck_head_centerpoint_custom.onnx')
        torch.onnx.export(
            pts_backbone_neck_head,
            (x, ),
            f=pth_onnx_backbone_neck_head,
            input_names=('spatial_features', ),
            output_names=tuple(model.pts_bbox_head.output_names),
            dynamic_axes={
                name: {
                    0: 'batch_size',
                    2: 'H',
                    3: 'W'
                }
                for name in ['spatial_features'] +
                model.pts_bbox_head.output_names
            },
            verbose=False,
            opset_version=11,
        )
        print(f'Saved pts_backbone_neck_head onnx model:'
              f' {pth_onnx_backbone_neck_head}')


@MODELS.register_module()
class PillarFeatureNetONNX(PillarFeatureNetAutoware):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_input_features(self, features: torch.Tensor,
                           num_points: torch.Tensor,
                           coors: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.
        Returns:
            torch.Tensor: Features of pillars.
        """

        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = features[:, :, :3].sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                    -1, 1, 1)
            f_cluster = features[:, :, :3] - points_mean
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        dtype = features.dtype
        if self._with_voxel_center:
            center_feature_size = 3 if self.use_voxel_center_z else 2
            if not self.legacy:
                f_center = torch.zeros_like(
                    features[:, :, :center_feature_size])
                f_center[:, :, 0] = features[:, :, 0] - (
                    coors[:, 3].to(dtype).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = features[:, :, 1] - (
                    coors[:, 2].to(dtype).unsqueeze(1) * self.vy +
                    self.y_offset)
                if self.use_voxel_center_z:
                    f_center[:, :, 2] = features[:, :, 2] - (
                        coors[:, 1].to(dtype).unsqueeze(1) * self.vz +
                        self.z_offset)
            else:
                f_center = features[:, :, :center_feature_size]
                f_center[:, :, 0] = f_center[:, :, 0] - (
                    coors[:, 3].type_as(features).unsqueeze(1) * self.vx +
                    self.x_offset)
                f_center[:, :, 1] = f_center[:, :, 1] - (
                    coors[:, 2].type_as(features).unsqueeze(1) * self.vy +
                    self.y_offset)
                if self.use_voxel_center_z:
                    f_center[:, :, 2] = f_center[:, :, 2] - (
                        coors[:, 1].type_as(features).unsqueeze(1) * self.vz +
                        self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)

        features = torch.cat(features_ls, dim=-1)

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        return features

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            features (torch.Tensor): Point features in shape (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor):
        Returns:
            torch.Tensor: Features of pillars.
        """

        for pfn in self.pfn_layers:
            features = pfn(features)

        return features


@MODELS.register_module()
class SeparateHeadONNX(SeparateHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Order output's of the heads
        rot_heads = {k: None for k in self.heads.keys() if 'rot' in k}
        self.heads: Dict[str, None] = {
            'heatmap': None,
            'reg': None,
            'height': None,
            'dim': None,
            **rot_heads,
            'vel': None,
        }


@MODELS.register_module()
class CenterHeadONNX(CenterHead):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.task_heads: List[SeparateHeadONNX]
        self.output_names: List[str] = list(self.task_heads[0].heads.keys())

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Args:
            x (List[torch.Tensor]): multi-level features
        Returns:
            pred (Tuple[torch.Tensor]): Output results for tasks.
        """
        assert len(
            x
        ) == 1, 'The input of CenterHeadONNX must be a single-level feature'

        x = self.shared_conv(x[0])
        head_tensors: Dict[str, torch.Tensor] = self.task_heads[0](x)

        ret_list: List[torch.Tensor] = list()
        for head_name in self.output_names:
            ret_list.append(head_tensors[head_name])

        return tuple(ret_list)


class CenterPointHeadONNX(torch.nn.Module):

    def __init__(self, backbone: torch.nn.Module, neck: torch.nn.Module,
                 bbox_head: torch.nn.Module):
        super(CenterPointHeadONNX, self).__init__()
        self.backbone: torch.nn.Module = backbone
        self.neck: torch.nn.Module = neck
        self.bbox_head: torch.nn.Module = bbox_head

    def forward(self, x: torch.Tensor) -> Tuple[List[Dict[str, torch.Tensor]]]:
        """
        Args:
            x (torch.Tensor): (B, C, H, W)
        Returns:
            tuple[list[dict[str, any]]]:
                (num_classes x [num_detect x
                {'reg', 'height', 'dim', 'rot', 'vel', 'heatmap'}])
        """
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        x = self.bbox_head(x)
        return x


CUSTOM_MODEL_REGISTRY = Registry('model', parent=MODELS)
CUSTOM_MODEL_REGISTRY.register_module(module=PillarFeatureNetONNX, force=True)
CUSTOM_MODEL_REGISTRY.register_module(module=CenterHeadONNX, force=True)
CUSTOM_MODEL_REGISTRY.register_module(module=SeparateHeadONNX, force=True)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.cfg)
    det = CenterPointToONNX(
        config=cfg, checkpoint_path=args.ckpt, output_path=args.work_dir)
    det.save_onnx()


if __name__ == '__main__':
    main()
