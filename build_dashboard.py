#!/usr/bin/env python3
"""
Simple visual dashboard — no stats jargon, just thumbnails + plain labels.
"""
import argparse, base64, json, os, sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path.home() / "pixel_rage_research"

ANCHORS = [
    ("cute wholesome moment",        "😊", "Милый трогательный момент"),
    ("shock reaction, surprised face","😱", "Шок-реакция, удивлённое лицо"),
    ("threat from behind, danger",   "👀", "Угроза сзади, опасность"),
    ("skeleton ambush, arrow attack","🏹", "Засада скелета, атака стрелами"),
    ("absurd funny moment",          "😂", "Абсурдный смешной момент"),
    ("explosion, chaos, destruction","💥", "Взрыв, хаос, разрушение"),
    ("mystery, hidden object",       "🔍", "Тайна, скрытый объект"),
    ("sad emotional moment",         "😢", "Грустный эмоциональный момент"),
    ("character in mortal danger",   "💀", "Персонаж в смертельной опасности"),
    ("diamond ore, treasure",        "💎", "Алмазная руда, сокровище"),
    ("creeper behind player",        "🟢", "Крипер за спиной игрока"),
    ("zombie horde, mass attack",    "🧟", "Орда зомби, массовая атака"),
    ("player falling into lava",     "🌋", "Игрок падает в лаву"),
    ("epic achievement, victory",    "🏆", "Эпичное достижение, победа"),
    ("peaceful calm scene",          "😴", "Спокойная мирная сцена"),
]
ANCHOR_TEXTS  = [a[0] for a in ANCHORS]
ANCHOR_EMOJIS = [a[1] for a in ANCHORS]
ANCHOR_RU     = [a[2] for a in ANCHORS]


def img_b64(path, size=200):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = size / max(w, h)
        if s < 1:
            img = img.resize((int(w*s), int(h*s)))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=75)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except:
        return ""

def fmt(n):
    n = int(n or 0)
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.0f}K"
    return str(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="dataset")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--demo", action="store_true",
                    help="Skip loading CLIP model — use cached text embeddings")
    args = ap.parse_args()

    embeds = np.load(ROOT / f"{args.inp}_embeds.npy")
    metas  = json.load(open(ROOT / f"{args.inp}_meta.json"))
    N = len(metas)
    print(f"Loaded {N} videos")

    vn = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)

    # Text embeddings: load cache or compute with CLIP
    cache_path     = ROOT / "clip_text_cache.npz"
    anchor_key     = "anchors"
    candidate_key  = "candidates"

    if args.demo and cache_path.exists():
        print("Demo mode: loading cached text embeddings (no CLIP model needed)")
        cache  = np.load(cache_path)
        tf     = cache[anchor_key].astype(np.float32)
        cand_feat = cache[candidate_key].astype(np.float32)
        proc   = None
        model  = None
    else:
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        with torch.no_grad():
            ti = proc(text=ANCHOR_TEXTS, return_tensors="pt", padding=True)
            tf = model.get_text_features(**ti)
            tf = (tf / tf.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        # Save cache for next time
        np.savez(cache_path, **{anchor_key: tf})
        print(f"Saved anchor cache → {cache_path}")

    sims = vn @ tf.T  # (N, 15)

    views = np.array([m.get("views") or 0 for m in metas], dtype=float)
    subs  = np.array([m.get("channel_subs") or 0 for m in metas], dtype=float)
    vs    = views / np.clip(subs, 1, None)
    log_vs = np.log10(np.clip(vs, 1e-3, None))

    top_anchor = sims.argmax(axis=1)
    order = np.argsort(-vs)
    top_idx = order[:args.top]
    bot_idx = order[-args.top:][::-1]

    # ── Emotion ranking ──────────────────────────────────────────────────
    # For each anchor: mean views/sub of videos where this is top-1 anchor
    rank = []
    for ai, (label, emoji, ru_label) in enumerate(ANCHORS):
        mask = top_anchor == ai
        if mask.sum() == 0:
            avg = 0.0; cnt = 0
        else:
            avg = float(vs[mask].mean())
            cnt = int(mask.sum())
        rank.append((avg, ai, label, emoji, cnt))
    rank.sort(reverse=True)

    # Normalise to 0-100 bar width
    max_avg = max(r[0] for r in rank) or 1
    def bar_pct(v): return max(4, int(v / max_avg * 100))

    # ── Thumbnail card ───────────────────────────────────────────────────
    def card(i):
        m   = metas[i]
        b64 = img_b64(m.get("frame_path",""))
        ai  = int(top_anchor[i])
        emo = ANCHORS[ai][1] + " " + ANCHORS[ai][0]
        top3 = sorted(range(len(ANCHOR_TEXTS)), key=lambda a: -sims[i,a])[:3]
        top3_html = " · ".join(f"{ANCHOR_EMOJIS[a]} {ANCHOR_RU[a]}" for a in top3)
        return f"""
        <a href="{m.get('url','')}" target="_blank" class="card">
          <img src="{b64}">
          <div class="info">
            <div class="title">{(m.get('title') or '')[:60]}</div>
            <div class="nums">
              <b>{fmt(views[i])}</b> views &nbsp;·&nbsp;
              <b>{fmt(subs[i])}</b> subs &nbsp;·&nbsp;
              <span class="mult">{vs[i]:.0f}×</span> size
            </div>
            <div class="emo">Top emotions: {top3_html}</div>
          </div>
        </a>"""

    top_cards = "\n".join(card(int(i)) for i in top_idx)
    bot_cards = "\n".join(card(int(i)) for i in bot_idx)

    # ── Ranking bars ─────────────────────────────────────────────────────
    rank_rows = ""
    for pos, (avg, ai, label, emoji, cnt) in enumerate(rank):
        color = "#27ae60" if pos < 5 else ("#e67e22" if pos < 10 else "#e74c3c")
        ru = ANCHOR_RU[ai]
        rank_rows += f"""
        <tr>
          <td class="pos">#{pos+1}</td>
          <td class="lbl">{emoji} {ru}</td>
          <td class="bar-cell">
            <div class="bar" style="width:{bar_pct(avg)}%;background:{color}"></div>
          </td>
          <td class="avg">{avg:.0f}× avg</td>
          <td class="cnt">{cnt} видео</td>
        </tr>"""

    # ── 2D PCA map directly on 512d CLIP vectors ─────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords2d = pca.fit_transform(vn)   # vn = normalized 512d CLIP embeddings

    # Label axes via correlation with anchor sims
    def pc_label(pc_idx):
        axis = coords2d[:, pc_idx]
        corrs = [float(np.corrcoef(sims[:, ai], axis)[0, 1]) for ai in range(len(ANCHORS))]
        pos = int(np.argmax(corrs));  neg = int(np.argmin(corrs))
        return ANCHOR_RU[neg], ANCHOR_RU[pos]

    x_neg, x_pos = pc_label(0)
    y_neg, y_pos = pc_label(1)

    # Auto-label clusters via KMeans + CLIP similarity in 512d space
    from sklearn.cluster import KMeans
    cluster_colors = ["#e74c3c","#3498db","#9b59b6","#1abc9c","#e67e22","#f39c12"]

    # Rich candidate descriptions — CLIP will pick the best match per cluster centroid
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
        "kpop music dance performance colorful",
        "emotional sad crying character closeup",
        "explosion fire chaos destruction action",
        "victory celebration triumphant pose",
        "minecraft survival building crafting",
        "zombie mob attack horde",
        "steve shocked surprised face reaction",
        "minecraft title text overlay dark background",
        "noob vs pro challenge comparison",
        "cute friendship wholesome group moment",
    ]

    if args.demo and cache_path.exists() and candidate_key in np.load(cache_path):
        cand_feat = np.load(cache_path)[candidate_key].astype(np.float32)
    elif model is not None:
        with torch.no_grad():
            cand_in = proc(text=CANDIDATES, return_tensors="pt", padding=True)
            cand_feat = model.get_text_features(**cand_in)
            cand_feat = (cand_feat / cand_feat.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)
        # Update cache with candidates too
        np.savez(cache_path, **{anchor_key: tf, candidate_key: cand_feat})
    else:
        # Fallback: no candidate labels in pure demo mode without cache
        cand_feat = tf  # use anchor embeddings as fallback

    n_clusters = 6
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
        # Centroid in 512d space → find closest CLIP text description
        centroid_512 = vn[mask].mean(axis=0)
        centroid_512 /= np.linalg.norm(centroid_512) + 1e-9
        cand_sims = centroid_512 @ cand_feat.T
        top3 = np.argsort(-cand_sims)[:3]
        label_lines = [f"#{i+1} {CANDIDATES[j]}" for i, j in enumerate(top3)]
        avg_vs = float(vs[mask].mean())
        cluster_labels.append((cx, cy, label_lines, int(mask.sum()), avg_vs, mask))

    # For virality: use views/sub for BOTH size and color
    log_vs_arr = np.log10(np.clip(vs, 1e-3, None))
    # Size = views/sub (viral = big, dead = small)
    sizes_map = 8 + 36 * (log_vs_arr - log_vs_arr.min()) / max(log_vs_arr.max() - log_vs_arr.min(), 1)

    # Hover with thumbnail
    hover_map = []
    for i, m in enumerate(metas):
        ai  = int(top_anchor[i])
        fp  = m.get("frame_path", "")
        b64 = img_b64(fp, size=130) if fp and Path(fp).exists() else ""
        thumb = f"<br><img src='{b64}'>" if b64 else ""
        hover_map.append(
            f"<b>{(m.get('title') or '')[:65]}</b><br>"
            f"{fmt(views[i])} views · {fmt(subs[i])} subs · <b>{vs[i]:.0f}×</b><br>"
            f"{ANCHOR_EMOJIS[ai]} {ANCHOR_RU[ai]}"
            f"{thumb}"
        )

    vmin, vmax = log_vs_arr.min(), log_vs_arr.max()
    x_mid = float(coords2d[:, 0].mean())
    map_fig = go.Figure()

    # Points
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

    # Cluster label annotations only (no extra scatter traces = no duplicates)
    for ki, (cx, cy, label_lines, cnt, avg_vs, mask) in enumerate(cluster_labels):
        col = cluster_colors[ki % len(cluster_colors)]
        vs_tag = f"avg {avg_vs:.0f}× views/sub"
        text_body = "<br>".join(label_lines) + f"<br><i>{cnt} видео · {vs_tag}</i>"
        map_fig.add_annotation(
            x=cx, y=cy,
            text=text_body,
            showarrow=True, arrowhead=0, arrowcolor=col, arrowwidth=1.5,
            ax=0, ay=-50,
            font=dict(size=9.5, color="#111"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=col, borderwidth=1.5, borderpad=4,
            align="left",
        )

    # Top-5 labels
    for rank_i, idx in enumerate(order[:5]):
        title_short = (metas[idx].get("title") or "")[:28]
        map_fig.add_annotation(
            x=coords2d[idx, 0], y=coords2d[idx, 1],
            text=f"⭐{rank_i+1} {title_short}",
            showarrow=True, arrowhead=2, arrowcolor="#555",
            ax=30 if coords2d[idx,0] < x_mid else -30, ay=-30,
            font=dict(size=9, color="#111"),
            bgcolor="rgba(255,252,200,0.95)",
            bordercolor="#bbb", borderwidth=1, borderpad=3,
        )

    map_fig.update_layout(
        template="plotly_white",
        height=660,
        font=dict(size=11, family="-apple-system, sans-serif"),
        margin=dict(l=30, r=120, t=20, b=50),
        hovermode="closest",
        xaxis=dict(
            title=dict(
                text=f"← {x_neg}   ·····   {x_pos} →",
                font=dict(size=11, color="#555")),
            showgrid=False, zeroline=False, showticklabels=False,
        ),
        yaxis=dict(
            title=dict(
                text=f"↑ {y_pos}   ·····   {y_neg} ↓",
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

    # ── HTML ─────────────────────────────────────────────────────────────
    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<title>Hook Map — Minecraft Shorts</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,sans-serif;background:#f0f2f5;color:#111;font-size:13px}}
.page{{max-width:1100px;margin:0 auto;padding:20px}}
h1{{font-size:20px;font-weight:700;margin-bottom:4px}}
.sub{{color:#666;font-size:12px;margin-bottom:24px}}
h2{{font-size:15px;font-weight:700;margin:28px 0 6px;padding-bottom:5px;border-bottom:2px solid #222}}
.desc{{color:#555;font-size:12px;margin-bottom:12px}}

/* gallery */
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:8px}}
.col-head{{font-size:13px;font-weight:700;margin-bottom:8px}}
.viral .col-head{{color:#27ae60}}
.dead  .col-head{{color:#e74c3c}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}
.card{{display:block;background:#fff;border-radius:6px;overflow:hidden;
       text-decoration:none;color:inherit;
       box-shadow:0 1px 3px rgba(0,0,0,.1);transition:transform .1s}}
.card:hover{{transform:translateY(-2px);box-shadow:0 4px 10px rgba(0,0,0,.15)}}
.card img{{width:100%;aspect-ratio:9/16;object-fit:cover;display:block;background:#ddd}}
.info{{padding:6px 8px}}
.title{{font-size:10.5px;font-weight:600;line-height:1.3;
        max-height:28px;overflow:hidden;margin-bottom:3px}}
.nums{{font-size:10px;color:#555;margin-bottom:2px}}
.mult{{font-weight:700;color:#27ae60}}
.dead .mult{{color:#e74c3c}}
.emo{{font-size:9.5px;color:#6c3483;font-style:italic;line-height:1.3}}

/* ranking table */
.rank-table{{width:100%;border-collapse:collapse;margin-top:8px}}
.rank-table td{{padding:5px 8px;vertical-align:middle}}
.rank-table tr:nth-child(even){{background:#fafafa}}
.rank-table tr:hover{{background:#f0f0f0}}
.pos{{color:#999;font-size:11px;width:30px}}
.lbl{{font-size:12.5px;width:230px}}
.bar-cell{{width:300px}}
.bar{{height:16px;border-radius:3px;transition:width .3s}}
.avg{{font-size:11px;font-weight:700;white-space:nowrap;padding-left:8px}}
.cnt{{font-size:10px;color:#999;white-space:nowrap}}

/* insight box */
.insight{{background:#fff;border-left:4px solid #27ae60;border-radius:4px;
          padding:12px 16px;margin:12px 0;font-size:12.5px;line-height:1.7;
          box-shadow:0 1px 2px rgba(0,0,0,.06)}}
.insight p{{margin:4px 0}}

footer{{text-align:center;color:#aaa;font-size:10.5px;margin-top:30px;padding-top:12px;
        border-top:1px solid #ddd}}
</style>
</head><body>
<div class="page">

<h1>Minecraft Shorts — Hook Emotion Map
  {'<span style="background:#f39c12;color:#fff;font-size:12px;padding:2px 8px;border-radius:3px;margin-left:10px;font-weight:500">⚠ ОЗНАКОМИТЕЛЬНО — {N} видео, нужно 300+</span>' if N < 200 else ''}
</h1>
<div class="sub">{N} видео · CLIP ViT-B/32 · первый кадр → близость к 15 эмоциям{'  ·  <b style="color:#e67e22">демо-версия, результаты ориентировочные</b>' if N < 200 else ''}</div>

<!-- INSIGHT BOX -->
<div class="insight">
  <p>📌 <b>Что смотрим:</b> первый кадр каждого видео — то что человек видит в ленте за 0.1 секунды до решения листать дальше или остановиться.</p>
  <p>🧠 <b>Как работает:</b> CLIP (нейросеть, обученная на 400M картинок) переводит каждый кадр в 512-мерный вектор. Похожие по стилю кадры = близкие векторы. Затем мы меряем близость каждого кадра к 15 типовым эмоциям.</p>
  <p>📊 <b>Рейтинг эмоций</b> — для каждой эмоции считаем среднее views ÷ subscribers у видео с этим стилем кадра. Зелёный = этот стиль работает, красный = не работает.</p>
  <p>🗺️ <b>2D карта внизу</b> — все кадры в визуальном пространстве. Близкие точки = похожие по стилю. Цвет/размер = виральность. Кольца = автоматические кластеры, подписи = что CLIP видит в каждом кластере.</p>
  <p>⚠️ <b>Ограничения:</b> {N} видео — сигнал ориентировочный. При 300+ видео результаты станут статистически значимыми. CTR и retention (главные метрики алгоритма) недоступны извне — используем views/sub как прокси.</p>
</div>

<!-- GALLERY -->
<h2>Виральные vs мёртвые — первые кадры</h2>
<div class="desc">Клик → открывает видео. Сортировка по views ÷ subs.</div>
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
<div class="desc">
  Для каждой эмоции: среднее views/sub у видео, где эта эмоция ближайшая.
  Зелёный = работает, оранжевый = нейтрально, красный = избегать.
</div>
<table class="rank-table">
{rank_rows}
</table>

<!-- 2D MAP -->
<h2>2D карта первых кадров — CLIP 512d → PCA</h2>
<div class="insight">
  <p>🧠 <b>Что делает CLIP:</b> нейросеть (обучена на 400M картинок) превращает каждый первый кадр в вектор из 512 чисел — это "визуальный отпечаток" кадра. Похожие по стилю кадры имеют близкие векторы.</p>
  <p>📐 <b>Что делает PCA:</b> сжимает 512 чисел до 2 — чтобы можно было нарисовать на плоскости. <b>Близкие точки = визуально похожие первые кадры.</b> Далёкие = разные по стилю.</p>
  <p>🎨 <b>Цвет и размер точки</b> = views ÷ subscribers. Жёлтый/большой = видео взорвалось относительно канала. Синий/маленький = мёртвое.</p>
  <p>🏷️ <b>Метки эмоций на карте</b> = центр кластера видео с этой эмоцией. Зелёная метка = эта эмоция в среднем работает лучше медианы. Серая = хуже.</p>
  <p>🔍 <b>Оси X и Y</b> — автоматически подписаны по корреляции: показывают какая эмоция тянет вправо/влево/вверх/вниз.</p>
  <p>⭐ <b>Цель:</b> найти жёлтые кластеры рядом с зелёными метками — там визуальный стиль первого кадра который коррелирует с виральностью. Hover на точку → превьюха кадра + stats. Клик → видео на YouTube.</p>
</div>
{map_html}

<div style="background:#fff;border-left:4px solid #3498db;border-radius:4px;padding:12px 16px;margin:24px 0;font-size:12.5px;line-height:1.7;box-shadow:0 1px 2px rgba(0,0,0,.06)">
  <p style="font-weight:700;font-size:13px;margin-bottom:6px">🚀 Что ещё можно делать с этим пайплайном</p>
  <p>🎬 <b>Покадровый анализ одного видео</b> — вместо одного первого кадра разбить видео на кадры каждую секунду (или каждые 0.5с) и построить карту эмоций по времени. Сразу видно в какой момент визуальный стиль меняется, где хук, где кульминация, где люди уходят.</p>
  <p>📉 <b>Наложить на retention-кривую</b> — если взять свои видео из YouTube Studio (retention по секундам) и совместить с CLIP-картой кадров, можно буквально увидеть: "на 4-й секунде кадр уходит в зону 'спокойная сцена' — и именно там люди начинают уходить".</p>
  <p>🎯 <b>Анализ хуков конкурентов</b> — взять топ-50 видео любого канала, разбить по кадрам, найти эмоциональный паттерн первых 3 секунд. Это и есть формула их хука.</p>
  <p>🔄 <b>A/B тест первых кадров</b> — загрузить два варианта начала видео, посмотреть куда они попадают на карте. Выбрать тот что ближе к виральным кластерам.</p>
  <p>📦 <b>Автоматический скоринг</b> — перед публикацией каждого видео прогонять первый кадр через модель и получать оценку "насколько этот кадр близок к виральным". Если score низкий — переснять.</p>
</div>
<footer>
  {N} видео · dataset: {args.inp}_*.npy/json ·
  Обновить: <code>python3 collect_dataset.py --n 200 && python3 build_dashboard.py</code>
</footer>
</div></body></html>"""

    out = ROOT / args.out
    out.write_text(html)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
