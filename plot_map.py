#!/usr/bin/env python3
"""
Plot 2D CLIP hook map — videos only, colored by top emotion anchor.
Text anchors are NOT projected to 2D (CLIP has a modality gap that makes
that meaningless). Instead each video's top anchor is computed in the
original 512d space and used as its category/color.

Usage:
  python3 plot_map.py              # uses dataset_*
  python3 plot_map.py --in run1    # uses run1_*
"""
import argparse
import base64
import json
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import umap
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path.home() / "pixel_rage_research"

EMOTION_ANCHORS = [
    "threat from behind, danger approaching character",
    "shock reaction, surprised face close-up",
    "peaceful calm scene, character relaxing",
    "explosion, chaos, destruction",
    "character in mortal danger, about to die",
    "absurd funny moment, impossible physics",
    "sad emotional moment, character crying",
    "epic achievement, victory pose with treasure",
    "mystery, hidden object, curiosity",
    "cute wholesome moment, friendship",
    "creeper behind player, imminent explosion",
    "diamond ore, treasure discovery",
    "skeleton ambush, arrow attack",
    "player falling into lava",
    "zombie horde, mass attack",
]

# Short labels for legend / hover
ANCHOR_SHORT = [a.split(",")[0] for a in EMOTION_ANCHORS]

# Distinct colors for categorical anchors
ANCHOR_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
    "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f",
]


def img_to_b64(path: Path, max_side: int = 140) -> str:
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = max_side / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="dataset")
    ap.add_argument("--out", type=str, default="map.html")
    ap.add_argument("--neighbors", type=int, default=10)
    ap.add_argument("--min-dist", type=float, default=0.2)
    ap.add_argument("--no-thumbs", action="store_true")
    args = ap.parse_args()

    embeds_path = ROOT / f"{args.inp}_embeds.npy"
    meta_path = ROOT / f"{args.inp}_meta.json"
    if not embeds_path.exists():
        print(f"Missing {embeds_path}")
        sys.exit(1)

    video_embeds = np.load(embeds_path)
    with open(meta_path) as f:
        metas = json.load(f)
    N = len(metas)
    print(f"Loaded {N} videos, embed shape {video_embeds.shape}")

    # --- Compute anchor similarity in ORIGINAL 512d space ---
    print("Encoding text anchors in CLIP space...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    with torch.no_grad():
        t_in = processor(text=EMOTION_ANCHORS, return_tensors="pt", padding=True)
        t_feat = model.get_text_features(**t_in)
        t_feat = (t_feat / t_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)

    # Normalize video embeds just in case
    v_norm = video_embeds / (np.linalg.norm(video_embeds, axis=1, keepdims=True) + 1e-9)
    sims = v_norm @ t_feat.T  # (N, A)
    top_anchor = sims.argmax(axis=1)  # (N,)

    # --- UMAP on videos only ---
    n_neigh = min(args.neighbors, max(2, N - 1))
    print(f"Running UMAP (n_neighbors={n_neigh}, min_dist={args.min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neigh,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(v_norm)

    # --- Compute stats for coloring/sizing ---
    views = np.array([m.get("views", 0) or 0 for m in metas], dtype=float)
    subs = np.array([m.get("channel_subs", 0) or 0 for m in metas], dtype=float)
    subs_safe = np.clip(subs, 1, None)
    views_per_sub = views / subs_safe
    log_vs = np.log10(np.clip(views_per_sub, 1e-3, None))
    log_v = np.log10(np.clip(views, 1, None))

    if log_v.max() > log_v.min():
        sizes = 8 + 28 * (log_v - log_v.min()) / (log_v.max() - log_v.min())
    else:
        sizes = np.full_like(log_v, 14)

    # --- Hover text ---
    hover_texts = []
    for i, m in enumerate(metas):
        title = (m.get("title") or "")[:80]
        ch = m.get("channel") or ""
        v = int(m.get("views", 0) or 0)
        s = int(m.get("channel_subs", 0) or 0)
        vs = views_per_sub[i]
        emo = ANCHOR_SHORT[top_anchor[i]]
        emo_score = float(sims[i, top_anchor[i]])
        thumb_html = ""
        if not args.no_thumbs:
            fp = m.get("frame_path")
            if fp and Path(fp).exists():
                b64 = img_to_b64(Path(fp))
                if b64:
                    thumb_html = f"<br><img src='{b64}' width='140'>"
        hover = (
            f"<b>{title}</b><br>"
            f"channel: {ch}<br>"
            f"views: {v:,}<br>"
            f"subs:  {s:,}<br>"
            f"v/sub: {vs:.2f}<br>"
            f"top emotion: {emo} ({emo_score:.3f})"
            f"{thumb_html}"
        )
        hover_texts.append(hover)

    customdata = [[m.get("url", "")] for m in metas]

    # --- Build two traces: one per view mode, with visibility toggle ---
    fig = go.Figure()

    # Trace 1: colored by emotion (categorical, one trace per anchor for legend)
    for ai in range(len(EMOTION_ANCHORS)):
        mask = top_anchor == ai
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        fig.add_trace(go.Scatter(
            x=coords[idxs, 0],
            y=coords[idxs, 1],
            mode="markers",
            marker=dict(
                size=sizes[idxs],
                color=ANCHOR_COLORS[ai % len(ANCHOR_COLORS)],
                line=dict(width=1, color="white"),
                opacity=0.9,
            ),
            text=[hover_texts[i] for i in idxs],
            hoverinfo="text",
            name=ANCHOR_SHORT[ai],
            customdata=[customdata[i] for i in idxs],
            visible=True,
            legendgroup="emotion",
            legendgrouptitle_text="Top emotion (cosine in CLIP space)",
        ))

    n_emo_traces = len(fig.data)

    # Trace 2: colored by virality (views/sub) — single trace
    fig.add_trace(go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(
            size=sizes,
            color=log_vs,
            colorscale="Viridis",
            colorbar=dict(title="log10(views/sub)", x=1.02),
            showscale=True,
            cmin=log_vs.min(),
            cmax=log_vs.max(),
            line=dict(width=1, color="white"),
            opacity=0.9,
        ),
        text=hover_texts,
        hoverinfo="text",
        name="videos",
        customdata=customdata,
        visible=False,
    ))

    # Label top-5 by views/sub directly on the map
    top5_idx = np.argsort(-views_per_sub)[:5]
    for idx in top5_idx:
        title = (metas[idx].get("title") or "")[:35]
        fig.add_annotation(
            x=coords[idx, 0],
            y=coords[idx, 1],
            text=f"★ {title}",
            showarrow=True,
            arrowhead=2,
            ax=30,
            ay=-30,
            font=dict(size=10, color="black"),
            bgcolor="rgba(255,255,200,0.8)",
            bordercolor="black",
            borderwidth=1,
        )

    # --- Buttons to toggle mode ---
    emo_vis = [True] * n_emo_traces + [False]
    vir_vis = [False] * n_emo_traces + [True]
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.5, y=1.08, xanchor="center",
            buttons=[
                dict(label="Colored by top emotion",
                     method="update",
                     args=[{"visible": emo_vis},
                           {"title": "Minecraft Shorts hook map — colored by top emotion anchor"}]),
                dict(label="Colored by virality (views/sub)",
                     method="update",
                     args=[{"visible": vir_vis},
                           {"title": "Minecraft Shorts hook map — colored by log(views/sub)"}]),
            ],
        )],
        title=f"Minecraft Shorts hook map — colored by top emotion anchor ({N} videos)",
        xaxis_title="UMAP-1 (visual similarity)",
        yaxis_title="UMAP-2 (visual similarity)",
        width=1500,
        height=900,
        template="plotly_white",
        hovermode="closest",
        legend=dict(x=1.12, y=1, bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="gray", borderwidth=1),
    )

    # --- Explanatory text at bottom of HTML ---
    explainer = """
    <div style="max-width:1400px; margin:20px auto; padding:16px;
                background:#f4f4f4; border-radius:8px; font-family:sans-serif;">
      <h3 style="margin-top:0">How to read this map</h3>
      <ul>
        <li><b>Each dot = one Minecraft Short.</b> Position is set by UMAP
            on CLIP embeddings of the first frame — dots close together
            have visually similar first frames.</li>
        <li><b>Dot size</b> = log of total view count (bigger = more views).</li>
        <li><b>Dot color (default)</b> = which emotion-anchor is the closest
            match in the original 512d CLIP space. Anchors are NOT positioned
            on the map because text and image embeddings live in different
            regions of CLIP space (this is called the "modality gap" and
            would break the 2D projection).</li>
        <li><b>Toggle the top button</b> to switch to <i>virality</i> coloring:
            yellow = high views-per-subscriber (viral relative to channel size),
            purple = low.</li>
        <li><b>★ labels</b> mark the top-5 videos by views/sub.</li>
        <li><b>Hover</b> shows title, views, subs, and a thumbnail of the
            first frame. <b>Click</b> a point to open the video on YouTube.</li>
      </ul>
      <p><b>What to look for:</b> clusters of the same color = first frames
      with a shared emotional register. If a cluster has several large or
      yellow-coloured dots, that hook style is performing above-average.
      If a cluster is full of tiny dark dots, it's a saturated / overused look.</p>
    </div>
    """

    out_path = ROOT / args.out
    fig.write_html(
        out_path,
        include_plotlyjs="cdn",
        post_script="""
        var plot = document.querySelectorAll('.plotly-graph-div')[0];
        plot.on('plotly_click', function(data){
            var pt = data.points[0];
            if (pt.customdata && pt.customdata[0]) {
                window.open(pt.customdata[0], '_blank');
            }
        });
        """,
    )

    # Append explainer to HTML
    html = out_path.read_text()
    html = html.replace("</body>", explainer + "</body>")
    out_path.write_text(html)

    print(f"\nWrote {out_path}")
    print(f"Open: file://{out_path}")


if __name__ == "__main__":
    main()
