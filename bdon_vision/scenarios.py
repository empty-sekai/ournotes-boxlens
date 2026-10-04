"""Disjoint, versioned synthetic acquisition scenarios; no random-seed-only split."""
import io
import random
from PIL import Image, ImageFilter

VERSION='acquisition-v1'
PROFILES={
    'train':dict(resolutions=[(1280,720),(1600,900),(1920,1080),(1440,1080),(2160,1080)],
        ui_scale=(.82,1.10),resize=(.55,1.0),quality=[62,74,86,96],blur=(0,.45),rounds=[1,2]),
    'validation':dict(resolutions=[(1366,768),(1920,1200),(2340,1080),(2560,1440),(2048,1536)],
        ui_scale=(.88,1.04),resize=(.62,.94),quality=[57,69,81,91],blur=(.1,.65),rounds=[2,3]),
    'stress':dict(resolutions=[(960,540),(2280,1080),(2532,1170),(2800,1752),(3840,2160)],
        ui_scale=(.65,1.28),resize=(.30,.78),quality=[30,42,51,78],blur=(.2,1.15),rounds=[2,3,4]),
    # Held-back acquisition for a single evaluation after all models are frozen:
    # its screen sizes and quality steps appear in no other profile.
    'final':dict(resolutions=[(1334,750),(1792,828),(2224,1668),(2400,1080),(2388,1668)],
        ui_scale=(.86,1.06),resize=(.50,.92),quality=[47,64,77,88],blur=(.05,.8),rounds=[1,2,3]),
}


def acquire(scene,rng,profile):
    config=PROFILES[profile]
    original=scene.size
    ratio=rng.uniform(*config['resize'])
    size=tuple(max(64,round(v*ratio)) for v in original)
    scene=scene.convert('RGB').resize(size,rng.choice([Image.Resampling.BILINEAR,Image.Resampling.BICUBIC,Image.Resampling.LANCZOS]))
    sigma=rng.uniform(*config['blur'])
    scene=scene.filter(ImageFilter.GaussianBlur(sigma))
    stages=[]
    for _ in range(rng.choice(config['rounds'])):
        codec=rng.choice(['JPEG','WEBP']);quality=rng.choice(config['quality'])
        buf=io.BytesIO();scene.save(buf,format=codec,quality=quality)
        scene=Image.open(io.BytesIO(buf.getvalue())).convert('RGB')
        stages.append({'codec':codec,'quality':quality})
    # PNG preserves the already-applied acquisition artifacts without an
    # undocumented extra JPEG round during dataset serialization.
    return scene,dict(profile=profile,version=VERSION,source_resolution=original,
        output_resolution=scene.size,scale_x=scene.width/original[0],scale_y=scene.height/original[1],
        blur_sigma=sigma,compression=stages)
