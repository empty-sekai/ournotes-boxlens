"""Export a per-kind encoder to ONNX and build its gallery vectors.

Outputs in ``--output``:

* ``<kind>-encoder.onnx``: input ``image`` float32 [N, 3, H, W] RGB in [0, 1],
  output ``embedding`` float32 [N, 128], L2-normalised; opset 17, dynamic N.
* ``<kind>-gallery.json`` / ``<kind>-gallery.npz``: one vector per catalog
  card, each bound to the SHA-256 of the artwork bytes it was computed from.
* ``<kind>-export.json``: file hashes, operator list and consistency checks.

Gallery vectors are computed with the exported ONNX model on CPU, i.e. the
same graph a browser client runs.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .assets import read, write
from .kind_encoder import export_onnx, gallery_document, kind_cards, load_model, reference_source
from .kind_geometry import INPUT, MEMBER_CANONICAL, WINDOW, center_crop_box, reference_image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def alternative_reference(kind, path):
    """Reference input built with OpenCV area resampling instead of PIL.

    Used only to measure how sensitive gallery vectors are to a client's
    resampling implementation.
    """
    h, w = INPUT[kind]
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    if kind == 'member':
        if (rgb.shape[1], rgb.shape[0]) != MEMBER_CANONICAL:
            box = center_crop_box(rgb.shape[1], rgb.shape[0], MEMBER_CANONICAL[0] / MEMBER_CANONICAL[1])
        else:
            box = (0, 0, rgb.shape[1], rgb.shape[0])
    else:
        box = center_crop_box(rgb.shape[1], rgb.shape[0], WINDOW[kind][0] / WINDOW[kind][1])
    left, top, right, bottom = box
    matrix = np.array([[(right - left) / w, 0, left], [0, (bottom - top) / h, top]], np.float32)
    if (right - left) / w > 1.5:
        # Area-average to the target size first, as a downscaling client would.
        crop = rgb[int(round(top)):int(round(bottom)), int(round(left)):int(round(right))]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    else:
        out = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32).transpose(2, 0, 1) / 255.


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['member', 'snap'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--thresholds', type=Path, help='JSON with selected similarity/margin thresholds to embed')
    a = p.parse_args()
    import onnx
    import onnxruntime as ort

    kind = a.kind
    a.output.mkdir(parents=True, exist_ok=True)
    model = load_model(kind, a.model)
    onnx_path = a.output / f'{kind}-encoder.onnx'
    export_onnx(model, onnx_path, kind)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])

    catalog = read(a.data / 'catalog.json')['cards']
    cards = kind_cards(catalog, kind)
    sources = [reference_source(a.data, c) for c in cards]
    inputs = []
    for path in sources:
        with Image.open(path) as image:
            inputs.append(np.asarray(reference_image(kind, image), np.float32).transpose(2, 0, 1) / 255.)
    inputs = np.stack(inputs).astype(np.float32)
    vectors = np.concatenate([session.run(None, {'image': inputs[i:i + 16]})[0] for i in range(0, len(inputs), 16)])
    with torch.no_grad():
        reference = model(torch.from_numpy(inputs)).numpy()
    single = np.concatenate([session.run(None, {'image': inputs[i:i + 1]})[0] for i in range(len(inputs))])
    rng = np.random.default_rng(1)
    noise = rng.random((7, 3, *INPUT[kind]), dtype=np.float32)
    with torch.no_grad():
        noise_ref = model(torch.from_numpy(noise)).numpy()
    noise_onnx = session.run(None, {'image': noise})[0]
    alternative = np.stack([alternative_reference(kind, s) for s in sources]).astype(np.float32)
    alt_vectors = np.concatenate([session.run(None, {'image': alternative[i:i + 16]})[0]
                                  for i in range(0, len(alternative), 16)])
    alt_cos = (alt_vectors * vectors).sum(1)
    alt_rank = (alt_vectors @ vectors.T).argmax(1) == np.arange(len(vectors))
    checks = {
        'torch_vs_onnx_max_abs_diff': float(np.abs(reference - vectors).max()),
        'torch_vs_onnx_random_max_abs_diff': float(np.abs(noise_ref - noise_onnx).max()),
        'batch_vs_single_max_abs_diff': float(np.abs(single - vectors).max()),
        'vector_norm_range': [float(np.linalg.norm(vectors, axis=1).min()), float(np.linalg.norm(vectors, axis=1).max())],
        'alternative_resampling_cosine_min': float(alt_cos.min()),
        'alternative_resampling_cosine_mean': float(alt_cos.mean()),
        'alternative_resampling_self_retrieval': int(alt_rank.sum()), 'cards': len(cards),
    }
    model_sha = sha256(onnx_path)
    document = gallery_document(a.data, cards, kind, vectors, model_sha)
    if a.thresholds:
        document['thresholds'] = read(a.thresholds)
    gallery_json = a.output / f'{kind}-gallery.json'
    gallery_json.write_text(json.dumps(document, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    gallery_npz = a.output / f'{kind}-gallery.npz'
    np.savez(gallery_npz, ids=np.array([c['id'] for c in cards], np.int32), vectors=vectors.astype(np.float32),
             source_sha256=np.array([r['source_sha256'] for r in document['cards']]), model_sha256=np.array(model_sha))
    manifest = {'kind': kind, 'input_hw': list(INPUT[kind]), 'opset': 17, 'operators': operators,
                'files': {p.name: sha256(p) for p in [onnx_path, gallery_json, gallery_npz]},
                'source_state_dict_sha256': sha256(a.model), 'checks': checks}
    write(a.output / f'{kind}-export.json', manifest)
    print(json.dumps(manifest, indent=1), flush=True)


if __name__ == '__main__':
    main()
