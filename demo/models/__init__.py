from __future__ import absolute_import
from .mobilenet import MobileNet
from .resnet import ResNet34, ResNet50
from .resnet_vd import ResNet50_vd
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3_small_x1_0
from .pvanet import PVANet
__all__ = [
    "model_list", "MobileNet", "ResNet34", "ResNet50", "MobileNetV2", "PVANet",
    "ResNet50_vd", "MobileNetV3_small_x1_0"
]
model_list = [
    'MobileNet', 'ResNet34', 'ResNet50', 'MobileNetV2', 'PVANet',
    'ResNet50_vd', 'MobileNetV3_small_x1_0'
]
