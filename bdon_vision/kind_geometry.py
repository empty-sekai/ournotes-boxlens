"""Per-kind card geometry and encoder input preparation.

Member and Snap list tiles have different native layouts. Each kind gets an
encoder input with roughly its own artwork aspect ratio instead of a shared
portrait resize:

* member: tile 224x294 logical units, artwork window inset by 6 -> 212x282,
  encoder input 160x128 (H x W).
* snap: tile 326x184, artwork window 314x172, encoder input 128x224 (H x W).
  The 512x288 source artwork fills the window with a centered aspect crop.

Query crops come from a tile rectangle ``[x, y, w, h]`` in image pixels.
Reference inputs come from the catalog artwork with the same preprocessing a
browser client can reproduce (see ``reference_image``).
"""
import cv2
import numpy as np
from PIL import Image, ImageOps

KINDS = ('member', 'snap')
TILE = {'member': (224, 294), 'snap': (326, 184)}
ART_INSET = 6
WINDOW = {'member': (212, 282), 'snap': (314, 172)}
# Encoder input (height, width).
INPUT = {'member': (160, 128), 'snap': (128, 224)}
# Member references are built from the square sprite with an aspect fill.
MEMBER_CANONICAL = (212, 282)
# Normalized (x0, y0, x1, y1) regions inside the artwork window that the list
# UI draws over: attribute badge, level/field text and card-rank icon.
OVERLAYS = {
    'member': ((0., 0., .17, .13), (0., .82, .50, 1.), (.73, .80, 1., 1.)),
    'snap': ((0., 0., .11, .21), (0., .70, .31, 1.), (.84, .67, 1., 1.)),
}


def tile_scale(kind, bbox):
    """Image pixels per logical unit along x and y."""
    tw, th = TILE[kind]
    return bbox[2] / tw, bbox[3] / th


def art_window(kind, bbox):
    """Artwork window ``(left, top, width, height)`` for a tile rectangle."""
    x, y, w, h = (float(v) for v in bbox)
    sx, sy = tile_scale(kind, bbox)
    return x + ART_INSET * sx, y + ART_INSET * sy, w - 2 * ART_INSET * sx, h - 2 * ART_INSET * sy


def expand(rect, margin):
    """Grow ``(left, top, width, height)`` by ``margin`` of its size per side."""
    left, top, width, height = rect
    return left - width * margin, top - height * margin, width * (1 + 2 * margin), height * (1 + 2 * margin)


def inside_fraction(rect, image_size):
    """Fraction of a rectangle's area that lies inside an image ``(w, h)``."""
    left, top, width, height = rect
    iw, ih = image_size
    ox = max(0., min(iw, left + width) - max(0., left))
    oy = max(0., min(ih, top + height) - max(0., top))
    return ox * oy / max(width * height, 1e-9)


def sample_rect(image, rect, size, border=cv2.BORDER_REPLICATE, resample='linear'):
    """Resample ``rect`` from a BGR image into ``size`` (w, h); returns RGB.

    ``linear`` samples bilinearly at the output pixel centres. ``area`` first
    cuts the rectangle at source resolution and then area-averages it down,
    which is closer to what an image-smoothing canvas does when shrinking.
    """
    left, top, width, height = rect
    if resample == 'area' and (width > size[0] or height > size[1]):
        inner = (max(size[0], int(np.ceil(width))), max(size[1], int(np.ceil(height))))
        matrix = np.array([[width / inner[0], 0, left], [0, height / inner[1], top]], np.float32)
        cut = cv2.warpAffine(image, matrix, inner, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border)
        out = cv2.resize(cut, size, interpolation=cv2.INTER_AREA)
    else:
        matrix = np.array([[width / size[0], 0, left], [0, height / size[1], top]], np.float32)
        out = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=border)
    return np.ascontiguousarray(out[:, :, ::-1])


def query_crop(image, kind, bbox, margin=0., resample='linear'):
    """Encoder query crop (RGB uint8, H x W x 3) for a tile in a BGR screenshot.

    With ``margin > 0`` the window is expanded on every side and the output is
    scaled by ``1 + 2 * margin`` so the central region still has input size.
    ``resample`` is passed to :func:`sample_rect`.
    """
    h, w = INPUT[kind]
    rect = art_window(kind, bbox)
    if margin:
        rect = expand(rect, margin)
        w, h = round(w * (1 + 2 * margin)), round(h * (1 + 2 * margin))
    return sample_rect(image, rect, (w, h), resample=resample)


def center_crop_box(width, height, aspect):
    """Largest centered ``(left, top, right, bottom)`` box with ``aspect`` = w / h."""
    if width / height > aspect:
        cw = height * aspect
        left = (width - cw) / 2
        return left, 0., left + cw, float(height)
    ch = width / aspect
    top = (height - ch) / 2
    return 0., top, float(width), top + ch


def reference_image(kind, image):
    """Reference encoder input (PIL RGB, input size) from catalog artwork.

    member: the square sprite (or an already fitted 212x282 canonical image) is
    aspect-filled to 212x282 with LANCZOS, then resized to 128x160 bilinear.
    snap: the 512x288 artwork is center-cropped to the 314:172 window aspect,
    then resized to 224x128 bilinear.
    """
    image = image.convert('RGB')
    h, w = INPUT[kind]
    if kind == 'member':
        if image.size != MEMBER_CANONICAL:
            image = ImageOps.fit(image, MEMBER_CANONICAL, Image.Resampling.LANCZOS)
        return image.resize((w, h), Image.Resampling.BILINEAR)
    ww, wh = WINDOW[kind]
    box = center_crop_box(image.width, image.height, ww / wh)
    return image.resize((w, h), Image.Resampling.BILINEAR, box=box)


def cross_window(kind, source_kind):
    """Central ``(x, y)`` fraction of a ``source_kind`` artwork window that has
    the window aspect ratio of ``kind``."""
    target = WINDOW[kind][0] / WINDOW[kind][1]
    source = WINDOW[source_kind][0] / WINDOW[source_kind][1]
    return (target / source, 1.) if source > target else (1., source / target)


def cross_reference_image(kind, source_kind, image):
    """Reference input for ``kind`` cut from the window view of another kind's
    artwork (the same region :func:`cross_window` takes from a query crop)."""
    image = image.convert('RGB')
    if source_kind == 'member':
        if image.size != MEMBER_CANONICAL:
            image = ImageOps.fit(image, MEMBER_CANONICAL, Image.Resampling.LANCZOS)
        left, top, right, bottom = 0., 0., float(image.width), float(image.height)
    else:
        sw, sh = WINDOW[source_kind]
        left, top, right, bottom = center_crop_box(image.width, image.height, sw / sh)
    fx, fy = cross_window(kind, source_kind)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half_w, half_h = (right - left) * fx / 2, (bottom - top) * fy / 2
    h, w = INPUT[kind]
    return image.resize((w, h), Image.Resampling.BILINEAR, box=(cx - half_w, cy - half_h, cx + half_w, cy + half_h))


def to_chw(array):
    """uint8 H x W x 3 RGB -> float32 3 x H x W in [0, 1]."""
    return np.ascontiguousarray(np.asarray(array, np.float32).transpose(2, 0, 1) / 255.)


def overlay_mask(kind, height, width):
    """Boolean H x W mask of the UI overlay regions inside the artwork window."""
    mask = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in OVERLAYS[kind]:
        mask[int(round(y0 * height)):int(round(y1 * height)), int(round(x0 * width)):int(round(x1 * width))] = True
    return mask
