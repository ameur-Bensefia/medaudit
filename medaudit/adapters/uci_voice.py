"""UCI Parkinson's voice (Little et al.): 195 recordings, several per participant. Task: PD vs healthy.
Group: participant code embedded in `name` (phon_R01_S01_1 -> S01)."""
import os

import pandas as pd

from ._util import folder
from ..schema import Dataset


def load(root, **_):
    df = pd.read_csv(os.path.join(folder(root, "UCI_PD_Voice"), "parkinsons.data"))
    group = df["name"].str.extract(r"(S\d+)")[0].to_numpy()
    X = df.drop(columns=["name", "status"]).to_numpy(float)
    return Dataset("uci_voice", X, df["status"].to_numpy(), group, df["name"].to_numpy(),
                   task="Parkinson's disease vs healthy, voice recordings", notes=dict(source="UCI ML repository"))
