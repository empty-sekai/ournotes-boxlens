"""Per-kind open-set artwork encoders.

``KindEncoder`` keeps the MobileNetV3-Small feature extractor and the
``2x2 average pool -> Linear(576*4, 128) -> L2`` head of the shared encoder,
so both kinds can start from the same weights. The pooling window is chosen
per input size so the head always sees a 2x2 grid:

* member 160x128 -> 5x4 features, AvgPool kernel (3, 2) stride (2, 2)
* snap 128x224 -> 4x7 features, AvgPool kernel (2, 4) stride (2, 3)

Both are plain ONNX ``AveragePool`` operators.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .kind_geometry import INPUT, reference_image

POOL = {'member': ((3, 2), (2, 2)), 'snap': ((2, 4), (2, 3))}


def feature_grid(height, width):
    """MobileNetV3 feature map size (stride 32, 'same'-style padding)."""
    for _ in range(5):
        height, width = (height + 1) // 2, (width + 1) // 2
    return height, width


def pooled_grid(kind):
    (kh, kw), (sh, sw) = POOL[kind]
    fh, fw = feature_grid(*INPUT[kind])
    return (fh - kh) // sh + 1, (fw - kw) // sw + 1


def build_model(kind, pretrained=False):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class KindEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.kind = kind
            self.features = mobilenet_v3_small(weights=weights).features
            self.projection = nn.Linear(576 * 4, 128)
            self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
            self.pool_kernel, self.pool_stride = POOL[kind]

        def embed(self, x):
            x = self.features((x - self.mean) / self.std)
            x = F.avg_pool2d(x, self.pool_kernel, self.pool_stride).flatten(1)
            return self.projection(x)

        def forward(self, x):
            return F.normalize(self.embed(x), dim=1)

    return KindEncoder()


def load_model(kind, path):
    import torch
    model = build_model(kind)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    return model.eval()


def freeze_batchnorm(model):
    """Keep BatchNorm running statistics fixed (affine parameters still train)."""
    from torch import nn
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return model


def reference_array(kind, path):
    """float32 [3, H, W] reference input for a catalog artwork file."""
    with Image.open(path) as image:
        return np.asarray(reference_image(kind, image), np.float32).transpose(2, 0, 1) / 255.


def kind_cards(catalog, kind):
    return [c for c in catalog if c['kind'] == kind]


def reference_source(data, card):
    """Artwork file a reference is built from: member square sprite or snap thumbnail."""
    data = Path(data)
    if card['kind'] == 'member':
        square = data / f"native/assets/member-{card['asset_id']}-square.webp"
        if square.exists():
            return square
        return data / card.get('match_file', card['file'])
    return data / card['file']


def references(data, cards, kind):
    return np.stack([reference_array(kind, reference_source(data, c)) for c in cards])


def export_onnx(model, path, kind):
    import torch
    model = model.eval().cpu()
    h, w = INPUT[kind]
    torch.onnx.export(model, torch.zeros(1, 3, h, w), str(path), input_names=['image'], output_names=['embedding'],
                      dynamic_axes={'image': {0: 'batch'}, 'embedding': {0: 'batch'}}, opset_version=17, dynamo=False)


def gallery_document(data, cards, kind, vectors, model_sha256):
    """JSON-serialisable gallery bound to the artwork bytes each vector came from."""
    rows = []
    for card, vector in zip(cards, vectors):
        source = reference_source(data, card)
        rows.append({'kind': kind, 'id': card['id'], 'asset_id': card['asset_id'],
                     'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'vector': [round(float(v), 6) for v in vector]})
    h, w = INPUT[kind]
    return {'schema': 'bdon-kind-gallery/1', 'kind': kind, 'input_hw': [h, w], 'dimension': int(vectors.shape[1]),
            'model_sha256': model_sha256, 'cards': rows}
