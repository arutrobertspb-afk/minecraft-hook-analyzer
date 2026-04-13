#!/usr/bin/env python3
"""
Pixel Rage — Hook Map Dashboard
Generates a single self-contained HTML file with:
  - Viral vs Dead gallery (with 5-second emotion timeline per card)
  - Emotion ranking bar chart
  - 2D PCA map (CLIP 512d → 2D)

Usage:
  python3 build_dashboard.py               # uses dataset_*
  python3 build_dashboard.py --demo        # skip loading CLIP (use cache)
  python3 build_dashboard.py --in run1     # uses run1_*
"""
import argparse, base64, json, os, sys
from io import BytesIO
from pathlib import Path

# sklearn MUST be imported before torch on macOS/Anaconda to avoid OpenMP segfault
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path.home() / "pixel_rage_research"

ANCHORS = [
    ("cute wholesome moment",        "😊", "Милый момент"),
    ("shock reaction, surprised face","😱", "Шок-реакция"),
    ("threat from behind, danger",   "👀", "Угроза сзади"),
    ("skeleton ambush, arrow attack","🏹", "Засада скелета"),
    ("absurd funny moment",          "😂", "Абсурд/смех"),
    ("explosion, chaos, destruction","💥", "Взрыв/хаос"),
    ("mystery, hidden object",       "🔍", "Тайна/секрет"),
    ("sad emotional moment",         "😢", "Грусть"),
    ("character in mortal danger",   "💀", "Смертельная опасность"),
    ("diamond ore, treasure",        "💎", "Алмазы/сокровище"),
    ("creeper behind player",        "🟢", "Крипер сзади"),
    ("zombie horde, mass attack",    "🧟", "Орда зомби"),
    ("player falling into lava",     "🌋", "Падение в лаву"),
    ("epic achievement, victory",    "🏆", "Победа/достижение"),
    ("peaceful calm scene",          "😴", "Спокойная сцена"),
]
ANCHOR_TEXTS  = [a[0] for a in ANCHORS]
ANCHOR_EMOJIS = [a[1] for a in ANCHORS]
ANCHOR_SHORT  = [a[2] for a in ANCHORS]

FRAME_TIMES = [0.5, 1.5, 2.5, 3.5, 4.5]  # seconds

CANDIDATES = [
    "minecraft animation story with characters",
    "herobrine dark horror villain glowing eyes",
    "steve and alex cute couple romantic moment",
    "characters fighting combat battle scene",
    "parkour running jumping obstacle challenge",
    "baby cute small minecraft character",
    "evil villain laughing sinister face",
    "brainrot meme chaos random funny content",
    "minecraft horror scary dark night scene",
    "squid game lava jump challenge",
    "emotional sad crying character closeup",
    "explosion fire chaos destruction action",
    "victory celebration triumphant pose",
    "minecraft survival building crafting",
    "zombie mob attack horde",
    "steve shocked surprised face reaction",
    "minecraft title text overlay dark background",
    "noob vs pro challenge comparison",
    "cute friendship wholesome group moment",
    "tnt trap explosion surprise",
]


def img_b64(path, size=220):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = size / max(w, h)
        if s < 1:
            img = img.resize((int(w*s), int(h*s)))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=78)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""


def fmt(n):
    n = int(n or 0)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.0f}K"
    return str(n)


# Per-anchor colors for the timeline pills
ANCHOR_COLORS_HEX = [
    "#f9a825","#e53935","#7b1fa2","#1565c0","#2e7d32",
    "#bf360c","#37474f","#6a1a6a","#b71c1c","#00838f",
    "#558b2f","#4e342e","#e65100","#1b5e20","#78909c",
]


def timeline_html(frame_sims_5):
    """
    frame_sims_5: array shape (5, 15) — cosine similarity per frame per anchor.
    Returns HTML string for a 5-step timeline strip.
    """
    parts = []
    for t_i in range(frame_sims_5.shape[0]):
        top_a = int(frame_sims_5[t_i].argmax())
        emoji = ANCHOR_EMOJIS[top_a]
        label = ANCHOR_SHORT[top_a]
        color = ANCHOR_COLORS_HEX[top_a % len(ANCHOR_COLORS_HEX)]
        t_label = f"{FRAME_TIMES[t_i]:.1f}s"
        parts.append(
            f'<div class="tl-step" style="border-color:{color}" title="{t_label}: {label}">'
            f'<span class="tl-t">{t_label}</span>'
            f'<span class="tl-e">{emoji}</span>'
            f'<span class="tl-l">{label}</span>'
            f'</div>'
        )
    return '<div class="timeline">' + "".join(parts) + '</div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="dataset")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--demo", action="store_true",
                    help="Skip loading CLIP model — use cached text embeddings")
    args = ap.parse_args()

    embeds_path      = ROOT / f"{args.inp}_embeds.npy"
    frame_emb_path   = ROOT / f"{args.inp}_frame_embeds.npy"
    meta_path        = ROOT / f"{args.inp}_meta.json"

    if not embeds_path.exists():
        print(f"Missing {embeds_path}"); sys.exit(1)

    embeds = np.load(embeds_path)
    metas  = json.loads(meta_path.read_text())
    N = len(metas)
    print(f"Loaded {N} videos")

    # Per-frame embeddings (optional — enables timeline)
    has_timeline = frame_emb_path.exists()
    if has_timeline:
        frame_embeds = np.load(frame_emb_path)  # (N, 5, 512)
        print(f"  + frame_embeds shape {frame_embeds.shape}")
    else:
        print("  no frame_embeds — timeline disabled")
        frame_embeds = None

    vn = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)

    # ── Load or compute text embeddings ──────────────────────────────────
    cache_path    = ROOT / "clip_text_cache.npz"
    anchor_key    = "anchors"
    candidate_key = "candidates"

    if args.demo and cache_path.exists():
        print("Demo mode: loading cached text embeddings")
        cache     = np.load(cache_path)
        tf        = cache[anchor_key].astype(np.float32)
        cand_feat = cache.get(candidate_key, tf).astype(np.float32)
        model = proc = None
    else:
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        with torch.no_grad():
            ti = proc(text=ANCHOR_TEXTS, return_tensors="pt", padding=True)
            tf = model.get_text_features(**ti)
            tf = (tf / tf.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
            ci = proc(text=CANDIDATES, return_tensors="pt", padding=True)
            cand_feat = model.get_text_features(**ci)
            cand_feat = (cand_feat / cand_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        np.savez(cache_path, **{anchor_key: tf, candidate_key: cand_feat})
        print(f"Saved text cache → {cache_path}")

    # Similarities: (N, 15)
    sims = vn @ tf.T

    # Per-frame similarities for timeline: (N, 5, 15)
    if has_timeline:
        fn = frame_embeds / (np.linalg.norm(frame_embeds, axis=2, keepdims=True) + 1e-9)
        frame_sims = fn @ tf.T  # (N, 5, 15)
    else:
        frame_sims = None

    views = np.array([m.get("views") or 0 for m in metas], dtype=float)
    subs  = np.array([m.get("channel_subs") or 0 for m in metas], dtype=float)
    vs    = views / np.clip(subs, 1, None)
    log_vs = np.log10(np.clip(vs, 1e-3, None))

    top_anchor = sims.argmax(axis=1)
    order = np.argsort(-vs)
    top_idx = order[:args.top]
    bot_idx = order[-args.top:][::-1]

    # ── Emotion ranking ──────────────────────────────────────────────────
    rank = []
    for ai, (label, emoji, short) in enumerate(ANCHORS):
        mask = top_anchor == ai
        if mask.sum() == 0:
            avg = 0.0; cnt = 0
        else:
            avg = float(vs[mask].mean())
            cnt = int(mask.sum())
        rank.append((avg, ai, label, emoji, short, cnt))
    rank.sort(reverse=True)

    max_avg = max(r[0] for r in rank) or 1
    def bar_pct(v): return max(4, int(v / max_avg * 100))

    # ── Card builder ─────────────────────────────────────────────────────
    def card(i, show_mult_color="green"):
        m   = metas[i]
        fp  = m.get("frame_path", "")
        b64 = img_b64(fp)

        ai  = int(top_anchor[i])
        top3 = sorted(range(len(ANCHOR_TEXTS)), key=lambda a: -sims[i, a])[:3]
        top3_html = " · ".join(f"{ANCHOR_EMOJIS[a]} {ANCHOR_SHORT[a]}" for a in top3)

        # Timeline strip (5 seconds)
        tl = ""
        if frame_sims is not None:
            tl = timeline_html(frame_sims[i])

        mult_color = "#27ae60" if show_mult_color == "green" else "#e74c3c"
        rank_badge = f'<span class="rank-badge" style="background:{mult_color}">{vs[i]:.0f}×</span>'

        return f"""
        <a href="{m.get('url','')}" target="_blank" class="card">
          <div class="card-thumb">
            <img src="{b64}" loading="lazy">
            {rank_badge}
          </div>
          <div class="info">
            <div class="title">{(m.get('title') or '')[:65]}</div>
            <div class="nums">
              👁 <b>{fmt(views[i])}</b> &nbsp;·&nbsp;
              👥 <b>{fmt(subs[i])}</b> subs &nbsp;·&nbsp;
              ⏱ {m.get('duration',0)}s
            </div>
            <div class="emo">{top3_html}</div>
            {tl}
          </div>
        </a>"""

    top_cards = "\n".join(card(int(i), "green") for i in top_idx)
    bot_cards = "\n".join(card(int(i), "red") for i in bot_idx)

    # ── Ranking rows ─────────────────────────────────────────────────────
    rank_rows = ""
    for pos, (avg, ai, label, emoji, short, cnt) in enumerate(rank):
        color = "#27ae60" if pos < 5 else ("#e67e22" if pos < 10 else "#e74c3c")
        rank_rows += f"""
        <tr>
          <td class="pos">#{pos+1}</td>
          <td class="lbl">{emoji} {short}</td>
          <td class="bar-cell">
            <div class="bar" style="width:{bar_pct(avg)}%;background:{color}"></div>
          </td>
          <td class="avg">{avg:.0f}× avg</td>
          <td class="cnt">{cnt} видео</td>
        </tr>"""

    # ── 2D PCA map ───────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords2d = pca.fit_transform(vn)

    def pc_label(pc_idx):
        axis = coords2d[:, pc_idx]
        corrs = [float(np.corrcoef(sims[:, ai], axis)[0, 1]) for ai in range(len(ANCHORS))]
        pos = int(np.argmax(corrs)); neg = int(np.argmin(corrs))
        return ANCHOR_SHORT[neg], ANCHOR_SHORT[pos]

    x_neg, x_pos = pc_label(0)
    y_neg, y_pos = pc_label(1)

    cluster_colors = ["#e74c3c","#3498db","#9b59b6","#1abc9c","#e67e22","#f39c12"]

    n_clusters = min(6, max(2, N // 8))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(coords2d)
    ring_colors = [cluster_colors[k % len(cluster_colors)] for k in cluster_ids]

    cluster_labels = []
    for k in range(n_clusters):
        mask = cluster_ids == k
        if mask.sum() < 2:
            continue
        cx = float(coords2d[mask, 0].mean())
        cy = float(coords2d[mask, 1].mean())
        centroid_512 = vn[mask].mean(axis=0)
        centroid_512 /= np.linalg.norm(centroid_512) + 1e-9
        cand_sims = centroid_512 @ cand_feat.T
        top3 = np.argsort(-cand_sims)[:3]
        label_lines = [f"#{j+1} {CANDIDATES[idx]}" for j, idx in enumerate(top3)]
        avg_vs = float(vs[mask].mean())
        cluster_labels.append((cx, cy, label_lines, int(mask.sum()), avg_vs, mask))

    log_vs_arr = np.log10(np.clip(vs, 1e-3, None))
    vmin, vmax = log_vs_arr.min(), log_vs_arr.max()
    sizes_map = 8 + 36 * (log_vs_arr - vmin) / max(vmax - vmin, 1)
    x_mid = float(coords2d[:, 0].mean())

    hover_map = []
    for i, m in enumerate(metas):
        ai  = int(top_anchor[i])
        fp  = m.get("frame_path", "")
        b64 = img_b64(fp, size=130) if fp and Path(fp).exists() else ""
        thumb = f"<br><img src='{b64}'>" if b64 else ""
        tl_tip = ""
        if frame_sims is not None:
            tops = [ANCHOR_EMOJIS[int(frame_sims[i, t].argmax())] for t in range(frame_sims.shape[1])]
            tl_tip = f"<br>5s: {' → '.join(tops)}"
        hover_map.append(
            f"<b>{(m.get('title') or '')[:65]}</b><br>"
            f"{fmt(views[i])} views · {fmt(subs[i])} subs · <b>{vs[i]:.0f}×</b><br>"
            f"{ANCHOR_EMOJIS[ai]} {ANCHOR_SHORT[ai]}"
            f"{tl_tip}"
            f"{thumb}"
        )

    map_fig = go.Figure()
    map_fig.add_trace(go.Scatter(
        x=coords2d[:, 0], y=coords2d[:, 1],
        mode="markers",
        marker=dict(
            size=sizes_map,
            color=log_vs_arr,
            colorscale=[[0,"#3b4cc0"],[0.4,"#8e7cc3"],[0.65,"#f4a535"],[1,"#f7e52b"]],
            cmin=vmin, cmax=vmax,
            colorbar=dict(
                title=dict(text="views / subs", font=dict(size=11)),
                tickvals=[vmin, (vmin+vmax)/2, vmax],
                ticktext=["мало", "средне", "вирал"],
                thickness=14, len=0.55, x=1.01,
            ),
            showscale=True,
            line=dict(width=2, color=ring_colors),
            opacity=0.92,
        ),
        text=hover_map,
        hoverinfo="text",
        customdata=[[m.get("url","")] for m in metas],
    ))

    for ki, (cx, cy, label_lines, cnt, avg_vs, mask) in enumerate(cluster_labels):
        col = cluster_colors[ki % len(cluster_colors)]
        text_body = "<br>".join(label_lines) + f"<br><i>{cnt} видео · avg {avg_vs:.0f}×</i>"
        map_fig.add_annotation(
            x=cx, y=cy, text=text_body,
            showarrow=True, arrowhead=0, arrowcolor=col, arrowwidth=1.5,
            ax=0, ay=-55,
            font=dict(size=9.5, color="#111"),
            bgcolor="rgba(255,255,255,0.93)",
            bordercolor=col, borderwidth=1.5, borderpad=4,
            align="left",
        )

    for rank_i, idx in enumerate(order[:5]):
        title_short = (metas[idx].get("title") or "")[:28]
        map_fig.add_annotation(
            x=coords2d[idx, 0], y=coords2d[idx, 1],
            text=f"⭐{rank_i+1} {title_short}",
            showarrow=True, arrowhead=2, arrowcolor="#555",
            ax=30 if coords2d[idx, 0] < x_mid else -30, ay=-30,
            font=dict(size=9, color="#111"),
            bgcolor="rgba(255,252,200,0.95)",
            bordercolor="#bbb", borderwidth=1, borderpad=3,
        )

    map_fig.update_layout(
        template="plotly_white",
        height=680,
        font=dict(size=11, family="-apple-system, sans-serif"),
        margin=dict(l=30, r=130, t=20, b=50),
        hovermode="closest",
        xaxis=dict(
            title=dict(text=f"← {x_neg}   ·····   {x_pos} →",
                       font=dict(size=11, color="#555")),
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            title=dict(text=f"↑ {y_pos}   ·····   {y_neg} ↓",
                       font=dict(size=11, color="#555")),
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        plot_bgcolor="#fafafa",
    )

    map_html = pio.to_html(map_fig, include_plotlyjs="cdn", full_html=False,
                           post_script="""
        var p=document.querySelectorAll('.plotly-graph-div')[0];
        p.on('plotly_click',function(d){
            var pt=d.points[0];
            if(pt.customdata&&pt.customdata[0]) window.open(pt.customdata[0],'_blank');
        });
    """)

    has_tl_flag = "да" if has_timeline else "нет (нужно пересобрать датасет)"
    timeline_note = (
        "<p>⏱ <b>Таймлайн эмоций (5с)</b> = как меняется визуальный стиль за первые 5 секунд. "
        "Каждый кадр → ближайшая эмоция в 512-мерном CLIP-пространстве.</p>"
        if has_timeline else ""
    )

    # ── HTML ──────────────────────────────────────────────────────────────
    html = f"""<!doctype html><html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pixel Rage — Hook Map</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:#f0f2f5;color:#111;font-size:13px;line-height:1.5}}
.page{{max-width:1180px;margin:0 auto;padding:20px 16px}}

/* Header */
.hdr{{display:flex;align-items:baseline;gap:12px;margin-bottom:4px}}
h1{{font-size:22px;font-weight:800;letter-spacing:-.3px}}
.badge{{background:#f39c12;color:#fff;font-size:11px;padding:2px 9px;
        border-radius:12px;font-weight:600;flex-shrink:0}}
.sub{{color:#666;font-size:12px;margin-bottom:22px}}
h2{{font-size:15px;font-weight:700;margin:32px 0 6px;
    padding-bottom:6px;border-bottom:2px solid #222}}
.desc{{color:#555;font-size:12px;margin-bottom:14px}}

/* insight box */
.box{{background:#fff;border-left:4px solid #27ae60;border-radius:6px;
      padding:13px 16px;margin:14px 0 22px;font-size:12.5px;line-height:1.75;
      box-shadow:0 1px 4px rgba(0,0,0,.07)}}
.box p{{margin:3px 0}}
.box.blue{{border-color:#3498db}}

/* gallery */
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:10px}}
.col-head{{font-size:13px;font-weight:700;margin-bottom:10px;display:flex;
           align-items:center;gap:6px}}
.viral .col-head{{color:#27ae60}}
.dead  .col-head{{color:#e74c3c}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px}}

/* card */
.card{{display:block;background:#fff;border-radius:8px;overflow:hidden;
       text-decoration:none;color:inherit;
       box-shadow:0 1px 4px rgba(0,0,0,.10);transition:transform .12s,box-shadow .12s}}
.card:hover{{transform:translateY(-3px);box-shadow:0 6px 16px rgba(0,0,0,.15)}}
.card-thumb{{position:relative;overflow:hidden}}
.card img{{width:100%;aspect-ratio:9/16;object-fit:cover;display:block;background:#e8e8e8}}
.rank-badge{{position:absolute;top:6px;right:6px;background:#27ae60;color:#fff;
             font-size:11px;font-weight:700;padding:2px 7px;border-radius:10px;
             box-shadow:0 1px 3px rgba(0,0,0,.3)}}
.dead .rank-badge{{background:#e74c3c}}
.info{{padding:7px 9px 8px}}
.title{{font-size:10.5px;font-weight:600;line-height:1.35;
        max-height:30px;overflow:hidden;margin-bottom:4px;color:#111}}
.nums{{font-size:10px;color:#666;margin-bottom:3px}}
.emo{{font-size:9.5px;color:#6c3483;font-style:italic;line-height:1.4;margin-bottom:4px}}

/* 5-second timeline */
.timeline{{display:flex;gap:3px;margin-top:5px}}
.tl-step{{flex:1;border:1.5px solid #ccc;border-radius:4px;padding:2px 3px;
          text-align:center;cursor:default;transition:transform .1s}}
.tl-step:hover{{transform:scale(1.08)}}
.tl-t{{display:block;font-size:8px;color:#999;line-height:1}}
.tl-e{{display:block;font-size:13px;line-height:1.3}}
.tl-l{{display:block;font-size:7.5px;color:#555;line-height:1.2;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}

/* ranking table */
.rank-table{{width:100%;border-collapse:collapse;margin-top:8px}}
.rank-table td{{padding:5px 8px;vertical-align:middle}}
.rank-table tr:nth-child(even){{background:#fafafa}}
.rank-table tr:hover{{background:#f0f0f0}}
.pos{{color:#999;font-size:11px;width:32px}}
.lbl{{font-size:12.5px;width:200px}}
.bar-cell{{width:280px}}
.bar{{height:16px;border-radius:3px}}
.avg{{font-size:11px;font-weight:700;white-space:nowrap;padding-left:8px}}
.cnt{{font-size:10px;color:#999;white-space:nowrap}}

footer{{text-align:center;color:#aaa;font-size:10.5px;margin-top:32px;
        padding-top:12px;border-top:1px solid #ddd}}

@media(max-width:700px){{
  .pair{{grid-template-columns:1fr}}
  .grid{{grid-template-columns:repeat(2,1fr)}}
}}
</style>
</head><body>
<div class="page">

<div class="hdr">
  <h1>🎮 Pixel Rage — Hook Emotion Map</h1>
  {'<span class="badge">⚠ ДЕМО — ' + str(N) + ' видео</span>' if N < 200 else ''}
</div>
<div class="sub">
  {N} Minecraft Shorts · CLIP ViT-B/32 · первые 5 секунд → 15 эмоций
  {'· таймлайн: ' + has_tl_flag}
  {'&nbsp;&nbsp;<b style="color:#e67e22">Нужно 300+ для статистики</b>' if N < 200 else ''}
</div>

<div class="box">
  <p>📌 <b>Что смотрим:</b> первые 5 секунд каждого Short — то что зритель видит перед решением листать дальше или остановиться.</p>
  <p>🧠 <b>Как работает:</b> CLIP (нейросеть, 400M картинок) переводит каждый кадр в 512-мерный вектор. Мы берём кадры в 0.5, 1.5, 2.5, 3.5, 4.5с и измеряем близость к 15 эмоциям в оригинальном 512-мерном пространстве.</p>
  <p>📊 <b>Рейтинг эмоций</b> — для каждого эмоционального стиля: среднее views÷subscribers у видео с этим стилем. Зелёный = работает, красный = не работает.</p>
  {timeline_note}
  <p>🗺️ <b>2D карта</b> — PCA 512d→2D. Близкие точки = визуально похожие кадры. Цвет/размер = виральность (views/sub).</p>
  <p>⚠️ <b>Ограничения:</b> {N} видео — сигнал ориентировочный. CTR и retention недоступны снаружи, используем views/sub как прокси.</p>
</div>

<!-- GALLERY -->
<h2>Виральные vs мёртвые — первые кадры</h2>
<div class="desc">Клик → видео. Сортировка по views ÷ subs.</div>
<div class="pair">
  <div class="viral">
    <div class="col-head">✅ ТОП {args.top} — максимум views/sub</div>
    <div class="grid">{top_cards}</div>
  </div>
  <div class="dead">
    <div class="col-head">❌ БОТТОМ {args.top} — минимум views/sub</div>
    <div class="grid">{bot_cards}</div>
  </div>
</div>

<!-- RANKING -->
<h2>Рейтинг эмоций первого кадра — что реально работает</h2>
<div class="desc">Для каждой эмоции: среднее views/sub у видео с этим стилем.</div>
<table class="rank-table">
{rank_rows}
</table>

<!-- 2D MAP -->
<h2>2D карта первых кадров — CLIP 512d → PCA</h2>
<div class="box blue">
  <p>🧠 <b>PCA на 512-мерных CLIP-векторах:</b> близкие точки = визуально похожие первые кадры.</p>
  <p>🎨 <b>Цвет/размер точки</b> = views ÷ subs. Жёлтый/большой = вирал. Синий/маленький = мёртвое.</p>
  <p>🏷️ <b>Кольца вокруг точек</b> = авто-кластеры (KMeans). <b>Подписи</b> = что CLIP видит в центре кластера.</p>
  <p>⭐ <b>Звёзды</b> = топ-5 по views/sub. <b>Hover</b> = превью + 5-секундный таймлайн эмоций.</p>
</div>
{map_html}

<div class="box blue" style="margin-top:24px">
  <p style="font-weight:700;font-size:13px;margin-bottom:6px">🚀 Следующие шаги</p>
  <p>🎬 <b>Больше видео:</b> <code>python3 collect_dataset.py --n 300 --append</code> — добавить без перезаписи</p>
  <p>📉 <b>Retention overlay:</b> взять из YouTube Studio retention-кривые своих видео и наложить на CLIP-таймлайн — видно на какой секунде "эмоциональный спад" совпадает с уходом зрителей</p>
  <p>🎯 <b>Скоринг перед публикацией:</b> прогнать первый кадр через модель и получить оценку близости к виральным кластерам</p>
  <p>🔄 <b>A/B хуков:</b> сравнить два варианта первых секунд — куда они падают на 2D карте</p>
</div>

<footer>
  {N} видео · CLIP ViT-B/32 · таймлайн: {has_tl_flag} ·
  Обновить: <code>python3 collect_dataset.py --n 200 --append && python3 build_dashboard.py</code>
</footer>
</div></body></html>"""

    out = ROOT / args.out
    out.write_text(html)
    print(f"Wrote {out}")
    print(f"Open: file://{out}")


if __name__ == "__main__":
    main()
