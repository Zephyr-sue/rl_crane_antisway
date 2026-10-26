"""Interactive results dashboard: builds results/report.html (Plotly) from results/ and runs/.

    uv run viz/dashboard.py && start results/report.html
"""
import base64
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from utils.plotting import COLORS, ORDER, VARIANT_COLORS, VARIANT_RUNS, read_progress  # noqa: E402

RESULTS = "results"
LAYOUT = dict(template="plotly_white", font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#0b0b0b"),
              margin=dict(l=60, r=20, t=50, b=50), hovermode="x unified", height=380)


def learning_curves():
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validation return", "Validation success rate"))
    for label, pattern in VARIANT_RUNS.items():
        for k, f in enumerate(sorted(glob.glob(os.path.join("runs", pattern, "progress.csv")))):
            df = read_progress(f)
            if len(df) < 2:
                continue
            seed = os.path.basename(os.path.dirname(f)).split("seed")[-1]
            for col, metric in [(1, "val_return"), (2, "val_success")]:
                fig.add_trace(go.Scatter(x=df.env_steps, y=df[metric], name=f"GRPO {label} (seed {seed})",
                                         legendgroup=label, showlegend=(col == 1),
                                         line=dict(color=VARIANT_COLORS[label], width=2),
                                         opacity=1.0 if k == 0 else 0.55, hovertemplate="%{y:.2f}"), row=1, col=col)
    fig.update_xaxes(title_text="Environment steps")
    fig.update_layout(**LAYOUT, title="GRPO learning curves: main run and ablations (shared 64-scenario validation set)")
    return fig


def condition_bars(summary, metric, title):
    fig = go.Figure()
    for ctrl in [c for c in ORDER if c in set(summary.controller)]:
        sub = summary[summary.controller == ctrl]
        fig.add_trace(go.Bar(x=sub.condition, y=sub[metric], name=ctrl, marker_color=COLORS[ctrl],
                             hovertemplate=f"{ctrl}: %{{y:.2f}}<extra></extra>"))
    fig.update_layout(**{**LAYOUT, "hovermode": "closest"}, barmode="group", title=title)
    return fig


def heatmaps():
    path = os.path.join(RESULTS, "heatmap.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    names = [str(n) for n in data["names"]]
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=[f"{n}: success rate" for n in names],
                        shared_yaxes=True)
    for k, n in enumerate(names):
        fig.add_trace(go.Heatmap(z=data[f"g_{n}"], x=np.round(data["l"], 2), y=np.round(data["m2"], 2),
                                 colorscale="Blues", zmin=0, zmax=1, showscale=(k == len(names) - 1),
                                 hovertemplate="L1+L2=%{x} m<br>m2=%{y} kg<br>success=%{z:.2f}<extra></extra>"),
                      row=1, col=k + 1)
        fig.add_shape(type="rect", x0=1.0, x1=2.6, y0=0.5, y1=4.0, line=dict(dash="dash", color="#0b0b0b"),
                      row=1, col=k + 1)
    fig.update_xaxes(title_text="Total cable length L1+L2 (m)")
    fig.update_yaxes(title_text="Payload mass m2 (kg)", col=1)
    fig.update_layout(**{**LAYOUT, "hovermode": "closest"},
                      title="Robustness over payload and cable length (dashed box: training range)")
    return fig


def gif_tag(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return ""
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f'<img src="data:image/gif;base64,{b64}" style="max-width:100%;border-radius:8px">'


def main():
    summary = pd.read_csv(os.path.join(RESULTS, "summary.csv"))
    overall = summary.groupby("controller").mean(numeric_only=True)
    overall = overall.reindex([c for c in ORDER if c in overall.index]).round(2)

    ablation_html = ""
    if os.path.exists(os.path.join(RESULTS, "ablations.csv")):
        ablation_html = ("<section><h2>GRPO ablations (random scenarios)</h2>"
                         + pd.read_csv(os.path.join(RESULTS, "ablations.csv")).round(2).to_html(index=False, border=0)
                         + "</section>")
    figs = [learning_curves(),
            condition_bars(summary, "success", "Success rate by test condition"),
            condition_bars(summary, "settle_time", "Settling time (s) by test condition (lower is better)"),
            condition_bars(summary, "max_sway_deg", "Max sway angle (deg) by test condition"),
            heatmaps()]
    parts = [f.to_html(full_html=False, include_plotlyjs=("cdn" if i == 0 else False)) for i, f in enumerate(figs) if f]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crane Anti-Sway Results</title>
<style>
:root {{ --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e4e3df; }}
body {{ background:var(--bg); color:var(--ink); font-family:Inter,'Segoe UI',sans-serif; max-width:1100px;
        margin:0 auto; padding:24px 16px; }}
h1 {{ font-size:26px; margin-bottom:4px }} p.sub {{ color:var(--ink2); margin-top:0 }}
section {{ border-top:1px solid var(--line); padding:16px 0 }}
table {{ border-collapse:collapse; font-size:14px }} th, td {{ padding:6px 12px; border-bottom:1px solid var(--line); text-align:right }}
th:first-child, td:first-child {{ text-align:left }}
</style></head><body>
<h1>RL anti-sway control of a double-pendulum crane</h1>
<p class="sub">GRPO (critic-free, group-relative policy optimization) vs PPO (learned critic) vs the RBF neural-network adaptive controller of Qiang et al. (2021).</p>
<section><h2>Side-by-side rollout</h2>{gif_tag("compare.gif")}</section>
<section><h2>Overall (mean over all test conditions)</h2>{overall.to_html(classes="t", border=0)}</section>
{ablation_html}
{''.join(f'<section>{p}</section>' for p in parts)}
</body></html>"""
    out = os.path.join(RESULTS, "report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
