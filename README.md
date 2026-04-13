# Pixel Rage — Minecraft Shorts Hook Analyzer

Data-driven tool to find what first-frame visual styles correlate with virality on YouTube Shorts.

**How it works:**
1. Scrape Minecraft Shorts (yt-dlp + hashtag feeds)
2. Extract first 5 seconds (frames at 0.5, 1.5, 2.5, 3.5, 4.5s)
3. Embed each frame with CLIP ViT-B/32 (512-dim vectors)
4. Project to 2D with PCA, cluster with KMeans
5. Score proximity to 15 emotion anchors via cosine similarity in 512d space
6. Build interactive HTML dashboard with per-video emotion timelines

---

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. First CLIP download is ~600 MB.

---

## Usage

### 1. Collect dataset

```bash
# Collect 100 videos (saves dataset_embeds.npy, dataset_frame_embeds.npy, dataset_meta.json)
python3 collect_dataset.py --n 100

# Append more without overwriting existing data
python3 collect_dataset.py --n 100 --append

# Custom output name
python3 collect_dataset.py --n 50 --out run1
```

### 2. Build dashboard

```bash
# Full run (loads CLIP, ~30s)
python3 build_dashboard.py

# Demo mode — no CLIP needed, uses cached text embeddings
python3 build_dashboard.py --demo

# Open result
open ~/pixel_rage_research/dashboard.html
```

---

## Output files

| File | Description |
|------|-------------|
| `dataset_embeds.npy` | Shape (N, 512) — mean CLIP embedding per video |
| `dataset_frame_embeds.npy` | Shape (N, 5, 512) — per-frame embeddings for emotion timeline |
| `dataset_meta.json` | Video metadata (id, title, views, subs, url, frame_paths) |
| `clip_text_cache.npz` | Cached text embeddings — enables `--demo` mode |
| `dashboard.html` | Self-contained interactive dashboard |
| `frames/` | Extracted JPEG frames (frame_{idx}_t{0-4}.jpg) |

---

## Dashboard sections

- **Viral vs Dead gallery** — top/bottom 12 by views÷subscribers, with 5-second emotion timeline per card
- **Emotion ranking** — which emotional style correlates with virality
- **2D PCA map** — all videos in visual space, colored by virality (yellow = viral, blue = dead), with auto-labeled clusters

---

## Key metrics

- **views/sub ratio** — virality proxy: views ÷ max(subscribers, 1). Normalises for channel size so a 1M-view video on a 100K channel (10×) beats a 10M-view video on a 5M channel (2×).
- **Top emotion** — which of 15 emotion anchors has highest cosine similarity to the frame's CLIP vector in the original 512d space (not projected 2D — CLIP has a modality gap).
- **Emotion timeline** — top emotion at each of the 5 extracted frames, shows how visual style shifts through the first 5 seconds.

---

## Notes

- Dataset needs 300+ videos for statistically meaningful results
- CTR and retention are not accessible externally — views/sub is the best available proxy
- Strict shorts filter: duration ≤ 60s, vertical aspect ratio, ≥ 50 views, "minecraft" keyword in title or channel
