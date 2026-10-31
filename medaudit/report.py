"""Report card: one page summarizing the six modules."""
import html
import json

COLORS = {"green": "#1a9641", "amber": "#fdae61", "red": "#d7191c", "grey": "#9e9e9e", "inconclusive": "#7b68ee",
          "info": "#4a90d9"}
NAMES = {"A1": "Split integrity", "A2": "Identity and site shortcuts", "A3": "Data integrity",
         "A4": "Confounds", "A5": "Uncertainty", "A6": "Transfer"}


def verdict(results):
    r = [x["rating"] for x in results if x.get("rating") in ("green", "amber", "red", "inconclusive")]
    if "red" in r:
        return "red", "At least one audited threat was detected at the red level; see red modules."
    if "amber" in r:
        return "amber", "A threat was detected at a lower level; interpret with caution."
    if "inconclusive" in r:
        return "inconclusive", ("No threat was detected, but at least one could not be excluded with the available "
                                "data (insufficient evidence, not a detection).")
    return "green", "None of the audited threats was detected (this does not exclude threats outside the modules)."


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    return o


def to_json(results, meta, path):
    v, msg = verdict(results)
    with open(path, "w") as f:
        json.dump(_clean(dict(meta=meta, verdict=v, message=msg, modules=results)), f, indent=2, default=str)


def to_html(results, meta, path):
    v, msg = verdict(results)
    rows = "".join(
        f"<tr><td><b>{r['module']}</b> {NAMES.get(r['module'], '')}"
        f"{(' (' + html.escape(r['target']) + ')') if r.get('target') else ''}</td>"
        f"<td><span style='background:{COLORS.get(r['rating'], '#ccc')};color:white;padding:2px 10px;"
        f"border-radius:10px'>{r['rating']}</span></td><td>{html.escape(str(r.get('reason', '')))}</td></tr>"
        for r in results)
    m = "".join(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(val))}</li>" for k, val in meta.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Audit report card</title>
<style>body{{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#222}}
table{{border-collapse:collapse;width:100%}}td{{border-bottom:1px solid #ddd;padding:8px;vertical-align:top}}</style>
</head><body><h2>Audit report card</h2><ul>{m}</ul>
<p><span style='background:{COLORS[v]};color:white;padding:4px 14px;border-radius:12px'>Overall: {v}</span>
&nbsp;{html.escape(msg)}</p><table>{rows}</table>
<p style='font-size:12px;color:#666'>MedAudit report card. Ratings follow the default decision rules; see the
documentation for thresholds.</p></body></html>"""
    with open(path, "w") as f:
        f.write(page)
