"""Export a trained locator to ONNX and check it against PyTorch on CPU.

The exported graph takes ``image`` float32 [1, 3, H, W] (RGB, 0..1; H and W
multiples of 32) and returns ``heatmap`` [1, 2, H/8, W/8] (sigmoid),
``size`` [1, 2, H/8, W/8] (tile width and height in input pixels) and
``offset`` [1, 2, H/8, W/8] (tile center relative to the cell origin, in
cells). ``bdon_vision.locator.decode`` documents the decoding.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .locator import Locator, decode, letterbox, to_tensor
from .locator_model import ExportedLocator, LocatorNet, export_onnx


def compare(net, onnx_path, images, sizes, threshold=.3):
    import cv2
    model = ExportedLocator(net).eval().cpu()
    rows = []
    for path in images:
        image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        for size in sizes:
            locator = Locator(onnx_path, size, threads=1)
            canvas, scale = letterbox(image, size)
            with torch.no_grad():
                ref = [t[0].numpy() for t in model(torch.from_numpy(to_tensor(canvas)))]
            got = locator.raw(image)[:3]
            a, b = decode(*ref, scale, threshold), decode(*got, scale, threshold)
            same = len(a) == len(b) and all(x['kind'] == y['kind'] and np.allclose(x['bbox'], y['bbox'], atol=.05)
                                            for x, y in zip(a, b))
            rows.append({'image': Path(path).name, 'size': size, 'input': list(canvas.shape[:2]),
                         'max_abs_diff': {n: float(np.abs(r - g).max()) for n, r, g in zip(('heatmap', 'size', 'offset'), ref, got)},
                         'detections': len(a), 'decoded_identical': bool(same)})
    return rows


def benchmark(onnx_path, sizes, aspect=(4, 3), repeats=20, threads=1):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=['CPUExecutionProvider'])
    out = {}
    for size in sizes:
        h = -(-round(size * aspect[1] / aspect[0]) // 32) * 32
        x = np.random.default_rng(0).random((1, 3, h, size), dtype=np.float32)
        session.run(None, {'image': x})
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            session.run(None, {'image': x})
            times.append(time.perf_counter() - started)
        out[f'{size}x{h}'] = {'median_ms': 1000 * float(np.median(times)), 'min_ms': 1000 * float(np.min(times))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--width', type=int, default=96)
    p.add_argument('--images', nargs='*', default=[])
    p.add_argument('--sizes', type=int, nargs='+', default=[640])
    p.add_argument('--benchmark', action='store_true')
    a = p.parse_args()
    net = LocatorNet(width=a.width)
    net.load_state_dict(torch.load(a.weights, map_location='cpu', weights_only=True))
    net.eval()
    export_onnx(net, a.output)
    import onnx
    graph = onnx.load(a.output)
    onnx.checker.check_model(graph)
    report = {'onnx': a.output, 'sha256': hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),
              'bytes': Path(a.output).stat().st_size, 'opset': graph.opset_import[0].version,
              'operators': sorted({n.op_type for n in graph.graph.node}),
              'parameters': int(sum(t.numel() for t in net.parameters()))}
    if a.images:
        report['parity'] = compare(net, a.output, a.images, a.sizes)
    if a.benchmark:
        report['cpu_single_thread'] = benchmark(a.output, a.sizes)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
