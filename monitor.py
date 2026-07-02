"""Live training monitor for drone-jepa.

Tails the training log files in artifacts/ and live-plots the metrics. Pure log
parsing — no torch / model loading — so it is cheap and safe to run while
training is in progress.

Run with:   streamlit run monitor.py
(use a different port from app.py, e.g.  --server.port 8502)

It understands the log lines emitted by train.py / train_baseline.py, e.g.
  [s1   200] loss=0.67 pred=0.67 sig=0.014 latent_std=0.84      (stage 1)
  [s2   200] loss=0.61 pos=0.62 vel=1.66 rot=0.44 omega=4.26     (stage 2)
  [base  200] loss=0.40 pos=0.58 vel=1.46 rot=0.35 omega=2.93    (baseline)
"""

from __future__ import annotations

import glob
import os
import re
import time

import plotly.graph_objects as go
import streamlit as st

LOG_GLOB = "artifacts/train*.log"  # training logs only (skip data-collection logs)
LINE_RE = re.compile(r"\[(\w+)\s+(\d+)\]\s+(.*)")
KV_RE = re.compile(r"(\w+)=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
STAGE_NAMES = {"s1": "Stage 1 (latent)", "s2": "Stage 2 (prober)", "base": "Baseline"}
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

st.set_page_config(page_title="drone-jepa training monitor", layout="wide")
st.title("📊 drone-jepa — live training monitor")


def parse_log(path):
    """-> (series: {tag: {metric: [(step,val)]}}, headlines: [str], mtime)."""
    series, head = {}, []
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return series, head, None
    for ln in lines:
        m = LINE_RE.search(ln)
        if not m:
            s = ln.strip()
            if s and "=" not in s and not s.startswith("["):
                head.append(s)
            continue
        tag, step = m.group(1), int(m.group(2))
        d = series.setdefault(tag, {})
        for k, v in KV_RE.findall(m.group(3)):
            d.setdefault(k, []).append((step, float(v)))
    return series, head, os.path.getmtime(path)


def status_of(head, mtime):
    text = " ".join(head).lower()
    done = any(w in text for w in ("saved", "done", "exit=0"))
    fresh = mtime is not None and (time.time() - mtime) < 20
    if fresh:
        return "🟢 running", "running"
    if done:
        return "✅ finished", "done"
    return "⚪ idle", "idle"


def trainval_chart(series):
    """Overlay each tag's train metric (solid) vs its held-out val_ counterpart
    (dashed). A widening gap = overfitting. Picks pos>pred>loss as the metric."""
    fig = go.Figure()
    found = False
    for i, (tag, metrics) in enumerate(series.items()):
        base = next((m for m in ("pos", "pred", "loss") if m in metrics), None)
        if base is None:
            continue
        col = COLORS[i % len(COLORS)]
        xs, ys = zip(*metrics[base])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=col),
                                 name=f"{STAGE_NAMES.get(tag, tag)} · {base} (train)"))
        if f"val_{base}" in metrics:
            found = True
            vx, vy = zip(*metrics[f"val_{base}"])
            fig.add_trace(go.Scatter(x=vx, y=vy, mode="lines",
                                     line=dict(color=col, dash="dash"),
                                     name=f"{STAGE_NAMES.get(tag, tag)} · {base} (val)"))
    title = "train vs held-out val (gap = overfitting)" if found else "train metric"
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), title=title,
                      xaxis_title="step", legend=dict(orientation="h", y=-0.25))
    return fig


def line_chart(series, metric, title, logy=False):
    fig = go.Figure()
    for i, (tag, metrics) in enumerate(series.items()):
        if metric not in metrics:
            continue
        pts = metrics[metric]
        xs, ys = zip(*pts)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 name=STAGE_NAMES.get(tag, tag),
                                 line=dict(color=COLORS[i % len(COLORS)])))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0),
                      title=title, xaxis_title="step",
                      yaxis_type="log" if logy else "linear",
                      legend=dict(orientation="h", y=-0.25))
    return fig


# ---- sidebar: pick which logs to watch (default = recently modified) ------ #
all_logs = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime, reverse=True)
if not all_logs:
    st.warning(f"No log files found in {LOG_GLOB}. Start a training run "
               "(its stdout is redirected to artifacts/*.log).")
    st.stop()
recent = [p for p in all_logs if time.time() - os.path.getmtime(p) < 7200]
default = recent or all_logs[:3]
with st.sidebar:
    st.header("Logs")
    chosen = st.multiselect("Monitor these logs", all_logs, default=default,
                            format_func=os.path.basename)
    refresh = st.slider("Refresh every (s)", 1, 10, 2)
    logy = st.checkbox("Log scale for loss", True)
    st.caption("Auto-refreshes; charts update as training writes new lines.")


@st.fragment(run_every=refresh)
def render():
    st.caption(f"updated {time.strftime('%H:%M:%S')}")
    for path in chosen:
        series, head, mtime = parse_log(path)
        badge, _ = status_of(head, mtime)
        # latest step across tags
        last_step = max((pts[-1][0] for d in series.values() for pts in d.values()),
                        default=0)
        st.subheader(f"`{os.path.basename(path)}`  ·  {badge}  ·  step {last_step}")
        if not series:
            st.caption("waiting for first metrics line…")
            continue

        # latest metrics as a row of st.metric per tag
        cols = st.columns(max(1, len(series)))
        for col, (tag, metrics) in zip(cols, series.items()):
            with col:
                st.markdown(f"**{STAGE_NAMES.get(tag, tag)}**")
                latest = {k: v[-1][1] for k, v in metrics.items()}
                show = [k for k in ("loss", "pred", "latent_std", "pos", "vel",
                                    "sig", "omega") if k in latest]
                for k in show[:4]:
                    st.metric(k, f"{latest[k]:.4g}")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(line_chart(series, "loss", "loss", logy),
                            use_container_width=True, key=f"{path}-loss")
        with c2:
            st.plotly_chart(trainval_chart(series),
                            use_container_width=True, key=f"{path}-trainval")
        st.divider()


render()
