"""Hand-crafted features for physiological signals (kept simple and documented on purpose: the audit concerns
the evaluation, not the feature engineering)."""
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

EEG_BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45))


def band_powers(x, fs, bands=EEG_BANDS):
    """x: channels x samples. Returns log band power per channel and band, plus line length and variance."""
    f, p = welch(x, fs=fs, nperseg=min(x.shape[-1], int(2 * fs)), axis=-1)
    feats = [np.log(trapezoid(p[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=-1) + 1e-12) for lo, hi in bands]
    ll = np.log(np.mean(np.abs(np.diff(x, axis=-1)), axis=-1) + 1e-12)
    var = np.log(np.var(x, axis=-1) + 1e-12)
    return np.concatenate(feats + [ll, var])


def ecg_beat(signal, r, fs, half=0.25, n_points=60):
    """Morphology of one beat (resampled window around the R peak)."""
    h = int(half * fs)
    if r - h < 0 or r + h >= len(signal):
        return None
    w = signal[r - h:r + h]
    w = w - np.median(w)
    return np.interp(np.linspace(0, len(w) - 1, n_points), np.arange(len(w)), w)


def rr_features(r_locs, i, fs):
    """Pre-RR, post-RR, local mean RR (10 beats) and their ratios, in seconds."""
    pre = (r_locs[i] - r_locs[i - 1]) / fs if i > 0 else np.nan
    post = (r_locs[i + 1] - r_locs[i]) / fs if i + 1 < len(r_locs) else np.nan
    lo = max(1, i - 10)
    local = np.mean(np.diff(r_locs[lo - 1:i + 1])) / fs if i > 0 else np.nan
    return np.array([pre, post, local, pre / local if local else np.nan, post / local if local else np.nan])


def window_stats(x):
    """x: samples x channels -> mean, std, min, max, 10th/90th percentiles per channel."""
    return np.concatenate([np.nanmean(x, 0), np.nanstd(x, 0), np.nanmin(x, 0), np.nanmax(x, 0),
                           np.nanpercentile(x, 10, 0), np.nanpercentile(x, 90, 0)])


def lead_stats(x, fs):
    """x: samples x leads (ECG) -> per lead statistics and coarse spectral powers."""
    f, p = welch(x, fs=fs, nperseg=min(len(x), 256), axis=0)
    bands = [(0.5, 5), (5, 15), (15, 40)]
    sp = [np.log(trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=0) + 1e-12) for lo, hi in bands]
    return np.concatenate([window_stats(x)] + sp)
