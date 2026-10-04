"""CenterNet-style card-frame locator network (MobileNetV3-Large + light FPN)."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .locator import CLASSES, STRIDE


def _conv_bn(cin, cout, kernel=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _separable(channels):
    return nn.Sequential(_conv_bn(channels, channels, 3, groups=channels), _conv_bn(channels, channels, 1))


def _head(channels, outputs, bias=0.):
    layer = nn.Conv2d(channels, outputs, 1)
    nn.init.normal_(layer.weight, std=.01)
    nn.init.constant_(layer.bias, bias)
    return nn.Sequential(_conv_bn(channels, channels, 3, groups=channels), _conv_bn(channels, channels, 1), layer)


class LocatorNet(nn.Module):
    """Stride-8 dense predictor.

    ``forward`` returns raw training outputs: heatmap logits [N, C, h, w],
    log tile size in cells [N, 2, h, w] and center offset in cells
    [N, 2, h, w]. ``ExportedLocator`` converts them to the deployment format.
    """

    def __init__(self, width=96, backbone_weights=None):
        super().__init__()
        from torchvision.models import mobilenet_v3_large
        backbone = mobilenet_v3_large(weights=None)
        if backbone_weights:
            backbone.load_state_dict(torch.load(backbone_weights, map_location='cpu', weights_only=True))
        features = backbone.features
        self.stem, self.c3, self.c4, self.c5 = features[:4], features[4:7], features[7:13], features[13:16]
        self.l3, self.l4, self.l5 = _conv_bn(40, width), _conv_bn(112, width), _conv_bn(160, width)
        self.s4, self.s3 = _separable(width), _separable(width)
        self.context = nn.Sequential(_separable(width), _separable(width))
        self.heat = _head(width, len(CLASSES), bias=-math.log((1 - .05) / .05))
        self.size = _head(width, 2, bias=2.)
        self.offset = _head(width, 2, bias=.5)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = self.stem((x - self.mean) / self.std)
        c3 = self.c3(x)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        p4 = self.s4(self.l4(c4) + F.interpolate(self.l5(c5), scale_factor=2., mode='nearest'))
        p3 = self.s3(self.l3(c3) + F.interpolate(p4, scale_factor=2., mode='nearest'))
        p3 = p3 + self.context(p3)
        return self.heat(p3), self.size(p3), self.offset(p3)


class ExportedLocator(nn.Module):
    """Deployment outputs: sigmoid heatmap, tile size in input pixels, offset in cells."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        heat, size, offset = self.net(image)
        return torch.sigmoid(heat), torch.exp(size.clamp(max=8.)) * STRIDE, offset


def export_onnx(net, path, height=384, width=640, opset=17):
    model = ExportedLocator(net).eval().cpu()
    # Batch is fixed to one image; height and width are any multiple of 32.
    dims = {2: 'height', 3: 'width'}
    out_dims = {2: 'grid_height', 3: 'grid_width'}
    torch.onnx.export(model, torch.zeros(1, 3, height, width), str(path), input_names=['image'],
                      output_names=['heatmap', 'size', 'offset'],
                      dynamic_axes={'image': dims, 'heatmap': out_dims, 'size': out_dims, 'offset': out_dims},
                      opset_version=opset, dynamo=False)
