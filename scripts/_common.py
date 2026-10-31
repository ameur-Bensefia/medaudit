import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DATASETS = ["uci_voice", "darwin", "pamap2", "uci_har", "mitbih", "chbmit", "ptbxl", "ham10000", "isic2018_test",
            "nih_cxr", "chexpert", "abide", "hw_pahaw", "hw_newhandpdsig"]


def load_cfg(path=None):
    path = path or os.environ.get("MEDAUDIT_CONFIG", "configs/paths.yaml")
    cfg = yaml.safe_load(open(path))
    for k in ("output_dir", "cache_dir"):
        os.makedirs(cfg[k], exist_ok=True)
    return cfg


def out(cfg, *parts):
    p = os.path.join(cfg["output_dir"], *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def adapter_args(cfg, name):
    """Map a dataset name to (adapter, options)."""
    if name.startswith("hw_"):
        return "handwriting", dict(which="pahaw" if name == "hw_pahaw" else "newhandpd")
    opts = dict(cache_dir=cfg["cache_dir"])
    if name in ("nih_cxr", "chexpert"):
        opts["max_images"] = cfg.get("max_images_cxr", 20000)
    return name, opts
