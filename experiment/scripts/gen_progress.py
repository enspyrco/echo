#!/usr/bin/env python3
"""Render an in-progress BBH sweep JSONL into a self-contained live status page.

Usage: gen_progress.py <jsonl> <out.html> <start_epoch> <now_epoch>

Reads the streaming results file (meta line + one row per completed run),
computes overall + per-arm progress, pass rates, cost units, and a wall-clock
ETA, and writes a single auto-refreshing HTML file (no external assets).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

HAIKU_UNIT, SONNET_UNIT = 1, 3


def cost_units(arm: str, sub_calls: int) -> int:
    if arm == "haiku-only":
        return sub_calls * HAIKU_UNIT
    if arm == "sonnet-only":
        return sub_calls * SONNET_UNIT
    if arm == "echo-judge":
        return 3 * HAIKU_UNIT + (SONNET_UNIT if sub_calls > 3 else 0)
    if arm == "echo-oracle":
        return 2 * HAIKU_UNIT + (SONNET_UNIT if sub_calls > 2 else 0)
    return sub_calls * HAIKU_UNIT


def _fmt_dur(secs: float) -> str:
    secs = max(0, int(secs))
    h, m = secs // 3600, (secs % 3600) // 60
    if h:
        return f"{h}h {m}m"
    return f"{m}m {secs % 60}s"


def main() -> None:
    jsonl, out_html, start_epoch, now_epoch = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

    meta = {}
    rows = []
    try:
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("_meta"):
                    meta = r
                else:
                    rows.append(r)
    except FileNotFoundError:
        rows = []

    arms = meta.get("arms", [])
    subtasks = meta.get("subtasks", [])
    n_per = meta.get("n_per_subtask", 0)
    n_tasks = len(subtasks) * n_per if subtasks else 0
    total = n_tasks * len(arms) if arms else 0
    done = len(rows)

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    elapsed = now_epoch - start_epoch
    rate = done / elapsed if elapsed > 0 and done else 0  # runs/sec
    eta = (total - done) / rate if rate > 0 else 0
    pct = (100 * done / total) if total else 0

    # Per-arm rows
    arm_cards = []
    for arm in arms:
        rs = by_arm.get(arm, [])
        a_done = len(rs)
        a_total = n_tasks
        a_pass = sum(1 for r in rs if r["passed"])
        a_unp = sum(1 for r in rs if r["detail"] == "unparseable")
        thr = 3 if arm == "echo-judge" else 2
        a_esc = sum(1 for r in rs if r["sub_calls"] > thr)
        a_cost = sum(cost_units(arm, r["sub_calls"]) for r in rs)
        rate_s = f"{a_pass}/{a_done} ({a_pass / a_done:.3f})" if a_done else "-"
        a_pct = (100 * a_done / a_total) if a_total else 0
        arm_cards.append(f"""
      <div class="arm">
        <div class="arm-head"><span class="arm-name">{arm}</span><span class="arm-prog">{a_done}/{a_total}</span></div>
        <div class="bar"><div class="fill" style="width:{a_pct:.1f}%"></div></div>
        <div class="arm-stats">
          <span>pass <b>{rate_s}</b></span>
          <span>esc <b>{a_esc}</b></span>
          <span>cost <b>{a_cost}u</b></span>
          <span class="{'warn' if a_unp else ''}">unparseable <b>{a_unp}</b></span>
        </div>
      </div>""")

    status = "RUNNING" if done < total else "COMPLETE"
    status_cls = "running" if done < total else "complete"
    eta_str = "done" if done >= total else (_fmt_dur(eta) if rate > 0 else "estimating...")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="20">
<title>Echo BBH n={n_tasks} - live</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#0d1117; color:#e6edf3; margin:0; padding:2rem 1rem; }}
  .wrap {{ max-width:760px; margin:0 auto; }}
  h1 {{ font-size:1.15rem; font-weight:600; margin:0 0 .25rem; }}
  .sub {{ color:#7d8590; font-size:.8rem; margin-bottom:1.25rem; }}
  .badge {{ display:inline-block; padding:.15rem .55rem; border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.04em; }}
  .running {{ background:#1f6feb33; color:#58a6ff; }}
  .complete {{ background:#23863633; color:#3fb950; }}
  .overall {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:1.1rem 1.2rem; margin-bottom:1.25rem; }}
  .big {{ font-size:2rem; font-weight:700; }}
  .big small {{ font-size:1rem; color:#7d8590; font-weight:400; }}
  .bar {{ height:9px; background:#21262d; border-radius:6px; overflow:hidden; margin:.6rem 0; }}
  .fill {{ height:100%; background:linear-gradient(90deg,#1f6feb,#3fb950); transition:width .4s; }}
  .meta {{ display:flex; gap:1.5rem; flex-wrap:wrap; color:#7d8590; font-size:.8rem; }}
  .meta b {{ color:#e6edf3; }}
  .arm {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:.7rem .9rem; margin-bottom:.6rem; }}
  .arm-head {{ display:flex; justify-content:space-between; align-items:center; }}
  .arm-name {{ font-weight:600; }}
  .arm-prog {{ color:#7d8590; font-size:.8rem; }}
  .arm-stats {{ display:flex; gap:1.1rem; flex-wrap:wrap; font-size:.78rem; color:#7d8590; margin-top:.35rem; }}
  .arm-stats b {{ color:#e6edf3; }}
  .warn b {{ color:#d29922; }}
  .foot {{ color:#484f58; font-size:.72rem; margin-top:1.5rem; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Echo - cheap-routing on BBH <span class="badge {status_cls}">{status}</span></h1>
  <div class="sub">n={n_tasks} tasks &middot; {', '.join(subtasks)} &middot; Claude Max (zero API spend)</div>

  <div class="overall">
    <div class="big">{done}<small> / {total} runs</small> &nbsp; <small>{pct:.1f}%</small></div>
    <div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
    <div class="meta">
      <span>elapsed <b>{_fmt_dur(elapsed)}</b></span>
      <span>eta <b>{eta_str}</b></span>
      <span>pace <b>{rate * 60:.1f}/min</b></span>
    </div>
  </div>

  {''.join(arm_cards)}

  <div class="foot">auto-refreshes every 20s &middot; streaming from the runner's JSONL &middot; updated each cycle</div>
</div>
</body>
</html>"""

    with open(out_html, "w") as f:
        f.write(html)
    print(f"wrote {out_html}: {done}/{total} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
