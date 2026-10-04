"""Card-frame locator: input preparation, training targets and output decoding.

The locator network predicts, on a stride-8 grid, one center heatmap per card
kind, the tile width and height in network-input pixels, and the sub-cell
offset of the tile center. It locates card tiles and their kind only; card
identity is resolved by a separate retrieval step.

This module depends on NumPy only, so decoding can be reused and tested
without a deep-learning framework. ``Locator`` runs an exported ONNX model
with onnxruntime.
"""
import numpy as np

CLASSES = ('member', 'snap')
# Tile width / height of each class in the native list layout.
TILE_ASPECT = {'member': 224 / 294, 'snap': 326 / 184}
STRIDE = 8
MULTIPLE = 32
PAD_VALUE = 128
# Gaussian radius factor relative to the tile size (TTFNet-style elliptical
# targets). The regression area uses the same ellipse.
GAUSSIAN_ALPHA = .54
REGRESSION_FLOOR = .05


def letterbox(image, long_edge, multiple=MULTIPLE, pad_value=PAD_VALUE):
    """Resize ``image`` (H, W, 3 uint8) so its long edge equals ``long_edge``.

    The resized image is placed at the top-left corner of a canvas whose sides
    are rounded up to ``multiple``. Returns ``(canvas, scale)``; an input pixel
    coordinate maps back to the original image by dividing by ``scale``.
    """
    import cv2
    h, w = image.shape[:2]
    scale = long_edge / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    ch, cw = -(-nh // multiple) * multiple, -(-nw // multiple) * multiple
    canvas = np.full((ch, cw, 3), pad_value, np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, scale


def to_tensor(canvas):
    """HWC uint8 RGB -> float32 [1, 3, H, W] in 0..1."""
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.


def _gaussian_ellipse(width, height, alpha=GAUSSIAN_ALPHA):
    return max(alpha * width / STRIDE / 6, .35), max(alpha * height / STRIDE / 6, .35)


def encode_targets(boxes, labels, ignore, shape, classes=len(CLASSES)):
    """Build dense training targets on the stride-8 grid.

    ``boxes`` are tile rectangles ``[x, y, w, h]`` in input pixels, ``labels``
    their class indices and ``ignore`` rectangles where background loss is not
    applied (for example tiles that are less than half visible). ``shape`` is
    the input ``(H, W)``.

    Returns a dict with ``heatmap`` [C, h, w], ``loss_mask`` [h, w] (1 where the
    heatmap loss applies), ``box`` [4, h, w] (target center x, center y, width,
    height in input pixels) and ``weight`` [h, w] (regression weight, each
    object's weights sum to one).
    """
    gh, gw = shape[0] // STRIDE, shape[1] // STRIDE
    heat = np.zeros((classes, gh, gw), np.float32)
    mask = np.ones((gh, gw), np.float32)
    box = np.zeros((4, gh, gw), np.float32)
    weight = np.zeros((gh, gw), np.float32)
    owner = np.zeros((gh, gw), np.float32)
    ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
    for x, y, w, h in np.asarray(ignore, np.float32).reshape(-1, 4):
        x0, y0 = int(np.floor(x / STRIDE)), int(np.floor(y / STRIDE))
        x1, y1 = int(np.ceil((x + w) / STRIDE)), int(np.ceil((y + h) / STRIDE))
        mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
    for (x, y, w, h), label in zip(np.asarray(boxes, np.float32).reshape(-1, 4), labels):
        if w <= 0 or h <= 0:
            continue
        cx, cy = x + w / 2, y + h / 2
        px = min(max(int(np.floor(cx / STRIDE)), 0), gw - 1)
        py = min(max(int(np.floor(cy / STRIDE)), 0), gh - 1)
        sx, sy = _gaussian_ellipse(w, h)
        g = np.exp(-((xs - px) ** 2 / (2 * sx * sx) + (ys - py) ** 2 / (2 * sy * sy)))
        g[g < .01] = 0
        heat[label] = np.maximum(heat[label], g)
        heat[label, py, px] = 1.
        inside = (xs + .5 >= x / STRIDE) & (xs + .5 <= (x + w) / STRIDE) & (ys + .5 >= y / STRIDE) & (ys + .5 <= (y + h) / STRIDE)
        region = (g > REGRESSION_FLOOR) & inside
        region[py, px] = True
        take = region & (g > owner)
        owner[take] = g[take]
        local = np.where(region, g, 0.)
        local /= local.sum()
        weight[take] = local[take]
        box[0][take], box[1][take], box[2][take], box[3][take] = cx, cy, w, h
    mask[heat.max(axis=0) > 0] = 1
    return {'heatmap': heat, 'loss_mask': mask, 'box': box, 'weight': weight}


def box_iou(a, b):
    """IoU matrix between ``[N, 4]`` and ``[M, 4]`` xywh rectangles."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    y1 = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    union = a[:, None, 2] * a[:, None, 3] + b[None, :, 2] * b[None, :, 3] - inter
    return inter / np.maximum(union, 1e-9)


def _overlap_smaller(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0., x1 - x0) * max(0., y1 - y0)
    return inter / max(min(a[2] * a[3], b[2] * b[3]), 1e-9)


def find_peaks(heatmap, threshold, max_peaks=300):
    """Local maxima (3x3) of a [C, h, w] heatmap at or above ``threshold``.

    Returns arrays ``(classes, rows, cols, scores)`` sorted by descending score.
    """
    c, h, w = heatmap.shape
    padded = np.pad(heatmap, ((0, 0), (1, 1), (1, 1)), constant_values=-1.)
    neighborhood = np.max(np.stack([padded[:, dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)]), axis=0)
    keep = (heatmap >= neighborhood) & (heatmap >= threshold)
    cls, rows, cols = np.nonzero(keep)
    scores = heatmap[cls, rows, cols]
    order = np.argsort(-scores, kind='stable')[:max_peaks]
    return cls[order], rows[order], cols[order], scores[order]


def visible_fraction(rect, image_size):
    """Share of an [x, y, w, h] rectangle inside a (width, height) image."""
    x, y, w, h = rect
    inside = max(0., min(image_size[0], x + w) - max(0., x)) * max(0., min(image_size[1], y + h) - max(0., y))
    return inside / max(w * h, 1e-9)


def decode(heatmap, size, offset, scale=1., threshold=.4, overlap=.5, max_detections=200,
           image_size=None, min_visible=.5, aspect_tolerance=None):
    """Turn raw network outputs (batch removed) into card detections.

    ``heatmap`` [C, h, w] is sigmoid-activated, ``size`` [2, h, w] holds tile
    width and height in input pixels, ``offset`` [2, h, w] holds the tile center
    relative to the cell origin in cells. Coordinates are divided by ``scale``
    to return to the original image.

    With ``image_size`` = (width, height), tiles less than ``min_visible``
    inside the image are dropped, matching the positive-tile definition. With
    ``aspect_tolerance`` t, tiles whose width / height differs from the class
    aspect by more than a factor 1 + t are dropped. Remaining detections whose
    intersection covers more than ``overlap`` of the smaller box are
    suppressed across classes, since card tiles never overlap on screen.
    """
    cls, rows, cols, scores = find_peaks(heatmap, threshold)
    out = []
    for k, i, j, s in zip(cls, rows, cols, scores):
        w, h = float(size[0, i, j]), float(size[1, i, j])
        cx, cy = (j + float(offset[0, i, j])) * STRIDE, (i + float(offset[1, i, j])) * STRIDE
        rect = [(cx - w / 2) / scale, (cy - h / 2) / scale, w / scale, h / scale]
        kind = CLASSES[int(k)]
        if image_size is not None and visible_fraction(rect, image_size) < min_visible:
            continue
        if aspect_tolerance is not None:
            ratio = w / max(h, 1e-9) / TILE_ASPECT[kind]
            if not 1 / (1 + aspect_tolerance) <= ratio <= 1 + aspect_tolerance:
                continue
        if any(_overlap_smaller(rect, d['bbox']) > overlap for d in out):
            continue
        out.append({'kind': kind, 'score': float(s), 'bbox': rect})
        if len(out) >= max_detections:
            break
    return out


class Locator:
    """CPU onnxruntime wrapper around an exported locator model."""

    def __init__(self, model, long_edge=640, threads=1, threshold=.4, aspect_tolerance=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        self.long_edge, self.threshold, self.aspect_tolerance = long_edge, threshold, aspect_tolerance

    def raw(self, image_rgb):
        canvas, scale = letterbox(image_rgb, self.long_edge)
        heat, size, offset = self.session.run(['heatmap', 'size', 'offset'], {'image': to_tensor(canvas)})
        return heat[0], size[0], offset[0], scale

    def __call__(self, image_rgb, threshold=None):
        heat, size, offset, scale = self.raw(image_rgb)
        return decode(heat, size, offset, scale, self.threshold if threshold is None else threshold,
                      image_size=image_rgb.shape[1::-1], aspect_tolerance=self.aspect_tolerance)
