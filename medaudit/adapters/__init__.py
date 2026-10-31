"""Dataset adapters. Each module exposes load(root, **options) -> medaudit.schema.Dataset."""
import importlib

ADAPTERS = ["uci_voice", "darwin", "pamap2", "mitbih", "chbmit", "ham10000", "nih_cxr", "chexpert", "abide",
            "handwriting", "ptbxl", "uci_har", "isic2018_test"]


def load(name, root, **opts):
    return importlib.import_module(f"medaudit.adapters.{name}").load(root, **opts)
