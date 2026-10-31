"""Features from raw pen signals (kinematics, pressure, pen lifts, spectral content).
Deliberately simple and documented: the audit concerns the evaluation, not the feature engineering."""
import numpy as np


def _stats(v, prefix):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_{k}": np.nan for k in ("mean", "std", "p10", "p50", "p90")}
    return {f"{prefix}_mean": v.mean(), f"{prefix}_std": v.std(), f"{prefix}_p10": np.percentile(v, 10),
            f"{prefix}_p50": np.percentile(v, 50), f"{prefix}_p90": np.percentile(v, 90)}


def tablet_features(x, y, t, pen_down, pressure, azimuth=None, altitude=None):
    """PaHaW-type tablet signals. t in seconds."""
    x, y, t, p = (np.asarray(a, float) for a in (x, y, t, pressure))
    down = np.asarray(pen_down) > 0
    dt = np.diff(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1e-2
    vx, vy = np.diff(x) / dt, np.diff(y) / dt
    speed = np.hypot(vx, vy)
    acc = np.diff(speed) / dt[1:]
    jerk = np.diff(acc) / dt[2:]
    seg_down = down[1:] & down[:-1]
    f = {"duration": t[-1] - t[0], "n_points": len(t),
         "in_air_ratio": float(1 - down.mean()),
         "pen_lifts": int(np.sum(np.diff(down.astype(int)) == -1)),
         "path_length_down": float(np.sum(np.hypot(np.diff(x), np.diff(y))[seg_down])),
         "width": np.ptp(x[down]) if down.any() else np.nan, "height": np.ptp(y[down]) if down.any() else np.nan}
    f.update(_stats(speed[seg_down], "speed_down"))
    f.update(_stats(speed[~seg_down], "speed_air"))
    f.update(_stats(np.abs(acc), "acc"))
    f.update(_stats(np.abs(jerk), "jerk"))
    f.update(_stats(p[down], "pressure"))
    f.update(_stats(np.diff(p)[seg_down] / dt[seg_down], "pressure_rate"))
    for name, a in (("azimuth", azimuth), ("altitude", altitude)):
        if a is not None:
            f.update(_stats(np.asarray(a, float)[down], name))
    if seg_down.sum() > 16:                      # tremor-band energy of the pen-down speed
        s = speed[seg_down] - speed[seg_down].mean()
        fs = 1.0 / np.median(dt)
        P = np.abs(np.fft.rfft(s)) ** 2
        fr = np.fft.rfftfreq(len(s), 1 / fs)
        tot = P[fr > 0.5].sum() + 1e-12
        f["tremor_4_12hz_ratio"] = float(P[(fr >= 4) & (fr <= 12)].sum() / tot)
    else:
        f["tremor_4_12hz_ratio"] = np.nan
    return f


def multichannel_features(M):
    """Biometric pen recordings (samples x channels): per-channel statistics, derivative statistics and
    low/high-frequency energy ratio."""
    M = np.asarray(M, float)
    f = {"n_samples": M.shape[0]}
    for j in range(M.shape[1]):
        c = M[:, j]
        f.update(_stats(c, f"ch{j}"))
        f.update(_stats(np.abs(np.diff(c)), f"ch{j}_d"))
        P = np.abs(np.fft.rfft(c - c.mean())) ** 2
        n = len(P)
        f[f"ch{j}_lowfreq_ratio"] = float(P[1:max(2, n // 10)].sum() / (P[1:].sum() + 1e-12))
    return f
