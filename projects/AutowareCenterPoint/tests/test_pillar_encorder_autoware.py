import pytest
import torch

from mmdet3d.registry import MODELS
from projects.AutowareCenterPoint.centerpoint.pillar_encoder_autoware import \
    PillarFeatureNetAutoware  # noqa: F401


def test_pillar_feature_net_autoware():

    use_voxel_center_z = False
    if not torch.cuda.is_available():
        pytest.skip('test requires GPU and torch+cuda')
    pillar_feature_net_autoware_cfg = dict(
        type='PillarFeatureNetAutoware',
        in_channels=4,
        feat_channels=[64],
        voxel_size=(0.2, 0.2, 8),
        point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        use_voxel_center_z=use_voxel_center_z,
        with_distance=False,
    )
    pillar_feature_net_autoware = MODELS.build(pillar_feature_net_autoware_cfg)

    features = torch.rand([97297, 20, 4])
    num_voxels = torch.randint(1, 100, [97297])
    coors = torch.randint(0, 100, [97297, 4])

    features = pillar_feature_net_autoware(features, num_voxels, coors)

    if not use_voxel_center_z:
        assert pillar_feature_net_autoware.pfn_layers[
            0].linear.in_features == 9
    else:
        assert pillar_feature_net_autoware.pfn_layers[
            0].linear.in_features == 9

    assert pillar_feature_net_autoware.pfn_layers[0].linear.out_features == 64

    assert features.shape == torch.Size([97297, 64])
