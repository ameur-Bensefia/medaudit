"""Frozen ImageNet ResNet-50 embeddings (2048-d), cached to disk. Uses Apple GPU (mps), CUDA or CPU.
Images can be file paths or (zip_path, member) tuples."""
import os
import zipfile

import numpy as np


def _device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _open(item, zips):
    from PIL import Image
    if isinstance(item, tuple):
        zp, member = item
        if zp not in zips:
            zips[zp] = zipfile.ZipFile(zp)
        with zips[zp].open(member) as f:
            return Image.open(f).convert("RGB")
    return Image.open(item).convert("RGB")


def embed(items, cache_path, batch=64, log_every=50):
    """Returns an (n, 2048) float32 array; resumes from a partial cache."""
    n = len(items)
    if os.environ.get("MEDAUDIT_FAKE_EMBED"):          # testing only: random features, no torch needed
        print("WARNING: MEDAUDIT_FAKE_EMBED is set -> RANDOM image features (testing only). "
              "Run: unset MEDAUDIT_FAKE_EMBED", flush=True)
        return np.random.default_rng(n).standard_normal((n, 2048)).astype(np.float32)
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    part = cache_path + ".partial.npy"
    if os.path.exists(cache_path):
        return np.load(cache_path)
    out = np.load(part) if os.path.exists(part) else np.full((n, 2048), np.nan, np.float32)
    start = int(np.argmax(np.isnan(out[:, 0]))) if np.isnan(out[:, 0]).any() else n
    dev = _device()
    net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    zips = {}
    with torch.no_grad():
        for b, i in enumerate(range(start, n, batch)):
            imgs = []
            for it in items[i:i + batch]:
                try:
                    imgs.append(tf(_open(it, zips)))
                except Exception:
                    imgs.append(torch.zeros(3, 224, 224))
            out[i:i + len(imgs)] = net(torch.stack(imgs).to(dev)).float().cpu().numpy()
            if b % log_every == 0:
                np.save(part, out)
                print(f"  embedded {min(i + batch, n)}/{n}", flush=True)
    np.save(cache_path, out)
    if os.path.exists(part):
        os.remove(part)
    return out
