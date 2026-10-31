"""Summary tables and figures from the outputs of scripts 01-04."""
import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from _common import load_cfg, out  # noqa: E402

cfg = load_cfg()
FIG = out(cfg, "figures", "x")[:-2]
COL = {"green": "#1a9641", "amber": "#fdae61", "red": "#d7191c", "grey": "#bdbdbd"}

# ---- Table: report-card matrix
p = out(cfg, "tables", "audit_summary.csv")
if os.path.exists(p):
    a = pd.read_csv(p)
    mods = [m for m in ["A1", "A2", "A3", "A4", "A5", "A6"] if m in a]
    a.to_csv(out(cfg, "tables", "table_report_cards.csv"), index=False)
    fig, ax = plt.subplots(figsize=(1.0 * len(mods) + 3, 0.32 * len(a) + 1))
    for i, r in a.reset_index(drop=True).iterrows():
        for j, m in enumerate(mods):
            ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, color=COL.get(str(r.get(m)), "#eeeeee")))
    ax.set_xlim(0, len(mods)); ax.set_ylim(0, len(a)); ax.invert_yaxis()
    ax.set_xticks([j + 0.47 for j in range(len(mods))]); ax.set_xticklabels(mods)
    ax.set_yticks([i + 0.45 for i in range(len(a))]); ax.set_yticklabels(a["dataset"] + " / " + a["model"], fontsize=7)
    ax.set_title("Audit report cards"); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_report_cards.pdf"))
    plt.close()
    # ---- Figure: forest plot of the participant-level gap
    d = a.dropna(subset=["group_delta"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(d) + 1))
    for i, r in d.iterrows():
        ax.errorbar(100 * r.group_delta, i, xerr=[[100 * (r.group_delta - r.group_delta_lo)],
                                                  [100 * (r.group_delta_hi - r.group_delta)]], fmt="o", color="k", ms=3)
    ax.axvline(0, color="grey", lw=0.8); ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["dataset"] + " / " + d["model"], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("Gap: sample-wise minus participant-wise balanced accuracy (points)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_gap_forest.pdf")); plt.close()

# ---- Figure: simulation flag rates
p = out(cfg, "simulation", "flag_rates.csv")
if os.path.exists(p):
    s = pd.read_csv(p)
    scen = sorted(s.scenario.unique())
    fig, axes = plt.subplots(1, len(scen), figsize=(3.2 * len(scen), 3), sharey=True)
    for ax, sc in zip(axes, scen):
        g = s[s.scenario == sc]
        for m, gm in g.groupby("module"):
            ax.plot(gm.level, gm.flag_rate, marker="o", label=m, lw=2 if gm.target.iloc[0] else 0.8,
                    ls="-" if gm.target.iloc[0] else ":")
        ax.set_title(sc.replace("_", " "), fontsize=9); ax.set_xlabel("injected strength"); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("flag rate"); axes[-1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_simulation.pdf")); plt.close()

# ---- Figure: predictive validity
p = out(cfg, "predictive", "configurations.csv")
if os.path.exists(p):
    c = pd.read_csv(p)
    pairs = [("site_bacc_internal", "drop"), ("a1_delta", "naive_optimism")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4))
    for ax, (x, y) in zip(axes, pairs):
        for ds_, g in c.groupby("dataset"):
            ax.scatter(100 * g[x], 100 * g[y], s=14, label=ds_, alpha=0.7)
        ax.axhline(0, color="grey", lw=0.6); ax.set_xlabel(x); ax.set_ylabel(y + " (points)")
    axes[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_predictive_validity.pdf"))
    plt.close()
print("figures written to", FIG)
