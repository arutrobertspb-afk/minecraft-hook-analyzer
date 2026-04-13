#!/usr/bin/env python3
"""
Collect dataset: search YouTube Shorts, download, extract first frame,
compute CLIP embeddings. Saves to dataset.npz + dataset_meta.json.

Usage:
  python3 collect_dataset.py --n 100 --query "minecraft shorts"
  python3 collect_dataset.py --n 50 --query "minecraft shorts viral" --out run1
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import yt_dlp
import imageio_ffmpeg

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path.home() / "pixel_rage_research"
FRAMES_DIR = ROOT / "frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def search_shorts(sources, n_per_source: int):
    """
    Collect candidate video IDs from multiple sources.
    Each source can be:
      - a bare query string  -> uses ytsearchN:query
      - a hashtag (starts with #) -> uses https://www.youtube.com/hashtag/<tag>/shorts
      - a full URL starting with http
    Returns deduped list of entries (dicts with at least 'id').
    """
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "playlistend": n_per_source,
    }
    seen = set()
    out = []
    with yt_dlp.YoutubeDL(opts) as ydl:
        for src in sources:
            src = src.strip()
            if src.startswith("http"):
                target = src
                label = src
            elif src.startswith("#"):
                tag = src.lstrip("#").replace(" ", "")
                target = f"https://www.youtube.com/hashtag/{tag}/shorts"
                label = f"hashtag:{tag}"
            else:
                target = f"ytsearch{n_per_source}:{src}"
                label = f"search:{src}"
            print(f"  source: {label}", flush=True)
            try:
                res = ydl.extract_info(target, download=False)
            except Exception as e:
                print(f"    failed: {e}", flush=True)
                continue
            entries = res.get("entries", []) or []
            kept = 0
            for e in entries[:n_per_source]:
                if not e:
                    continue
                vid = e.get("id")
                if not vid or vid in seen:
                    continue
                seen.add(vid)
                out.append(e)
                kept += 1
            print(f"    +{kept} new (total {len(out)})", flush=True)
    return out


def fetch_video(video_id: str, idx: int):
    """Download lowest quality + full metadata. Returns (video_path, meta_dict) or (None, None).
    Returns (None, 'skip:reason') if the video is filtered out for not being a short.
    """
    out_template = str(FRAMES_DIR / f"v_{idx}.%(ext)s")
    url = f"https://youtube.com/shorts/{video_id}"
    ydl_opts = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["ios", "android", "web"]}},
    }

    # First probe metadata without downloading to cheap-filter
    probe_opts = {**ydl_opts, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"  probe failed: {e}", flush=True)
        return None, None

    # STRICT FILTER: actual shorts only
    duration = info.get("duration") or 0
    if duration > 60 or duration <= 0:
        return None, f"skip:duration={duration}"

    width = info.get("width") or 0
    height = info.get("height") or 0
    if width and height and width >= height:
        return None, f"skip:landscape {width}x{height}"

    # Minimum view threshold to cut extreme noise (0 views = untested)
    vc = info.get("view_count") or 0
    if vc < 50:
        return None, f"skip:views={vc}"

    # Passed filters — actually download
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        print(f"  download failed: {e}", flush=True)
        return None, None

    video_path = None
    for ext in ("mp4", "webm", "mkv"):
        p = FRAMES_DIR / f"v_{idx}.{ext}"
        if p.exists():
            video_path = p
            break
    if video_path is None:
        return None, None

    meta = {
        "id": info.get("id"),
        "title": info.get("title") or "",
        "views": info.get("view_count") or 0,
        "likes": info.get("like_count") or 0,
        "duration": info.get("duration") or 0,
        "width": info.get("width") or 0,
        "height": info.get("height") or 0,
        "channel": info.get("channel") or "",
        "channel_subs": info.get("channel_follower_count") or 0,
        "upload_date": info.get("upload_date") or "",
        "url": f"https://youtube.com/shorts/{video_id}",
    }
    return video_path, meta


def extract_first_frame(video_path: Path, idx: int) -> Path:
    frame_path = FRAMES_DIR / f"frame_{idx}.jpg"
    cmd = [
        FFMPEG, "-y", "-ss", "0.1", "-i", str(video_path),
        "-vframes", "1", "-q:v", "2", str(frame_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return frame_path


KEYWORD_RE = re.compile(r"minecraft|steve|creeper|enderman|mcyt", re.IGNORECASE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Target number of videos to collect")
    ap.add_argument("--sources", type=str, nargs="+",
                    default=[
                        # Hashtag shorts feeds — these return ACTUAL shorts, not search-tagged videos
                        "#minecraft",
                        "#minecraftshorts",
                        "#minecraftmemes",
                        "#minecraftbuild",
                        "#minecraftpvp",
                        "#minecraftparkour",
                        "#minecraftsurvival",
                        "#minecraftanimation",
                        "#minecraftfunny",
                        "#mcyt",
                        # Fallback searches
                        "minecraft shorts",
                        "minecraft shorts viral",
                        "minecraft shorts funny",
                        "minecraft creeper shorts",
                        "minecraft steve shorts",
                    ],
                    help="Sources: hashtag (#tag), search query, or full URL")
    ap.add_argument("--per-source", type=int, default=40, help="Candidates per source")
    ap.add_argument("--out", type=str, default="dataset", help="Output basename (no ext)")
    ap.add_argument("--keyword-filter", action="store_true", default=True,
                    help="Require 'minecraft' in title/channel (default on)")
    ap.add_argument("--no-keyword-filter", dest="keyword_filter", action="store_false")
    args = ap.parse_args()

    print(f"Collecting from {len(args.sources)} sources, {args.per_source} each", flush=True)
    entries = search_shorts(args.sources, args.per_source)
    print(f"Got {len(entries)} unique candidates after dedupe", flush=True)

    print("Loading CLIP...", flush=True)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    embeds = []
    metas = []
    skipped = {"filter": 0, "keyword": 0, "download": 0, "frame": 0}
    for i, e in enumerate(entries):
        if len(metas) >= args.n:
            break
        vid = e.get("id")
        if not vid:
            continue
        print(f"[{len(metas)+1}/{args.n}] {vid} (tried {i+1})", flush=True)

        vpath, meta = fetch_video(vid, len(metas))
        if vpath is None:
            if isinstance(meta, str) and meta.startswith("skip:"):
                skipped["filter"] += 1
                print(f"  {meta}", flush=True)
            else:
                skipped["download"] += 1
            continue

        # Keyword filter: must actually be Minecraft
        if args.keyword_filter:
            blob = f"{meta.get('title','')} {meta.get('channel','')}"
            if not KEYWORD_RE.search(blob):
                skipped["keyword"] += 1
                print(f"  skip:keyword (title='{meta['title'][:40]}')", flush=True)
                try:
                    vpath.unlink()
                except Exception:
                    pass
                continue

        try:
            frame = extract_first_frame(vpath, len(metas))
            img = Image.open(frame).convert("RGB")
        except Exception as ex:
            print(f"  frame extract failed: {ex}", flush=True)
            skipped["frame"] += 1
            continue

        with torch.no_grad():
            img_in = processor(images=img, return_tensors="pt")
            feat = model.get_image_features(**img_in)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            vec = feat.squeeze(0).cpu().numpy().astype(np.float32)

        meta["frame_path"] = str(frame)
        embeds.append(vec)
        metas.append(meta)
        print(f"  OK  views={meta['views']:,} subs={meta['channel_subs']:,} "
              f"{meta['width']}x{meta['height']} | {meta['title'][:55]}", flush=True)

        # Delete video file to save disk (we only need the frame)
        try:
            vpath.unlink()
        except Exception:
            pass

    if not embeds:
        print("No videos collected.")
        sys.exit(1)

    embeds = np.stack(embeds, axis=0)
    np.save(ROOT / f"{args.out}_embeds.npy", embeds)
    with open(ROOT / f"{args.out}_meta.json", "w") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False)

    print(f"\nCollected {len(metas)} videos")
    print(f"  skipped: {skipped}")
    print(f"  embeds: {ROOT / (args.out + '_embeds.npy')}  shape={embeds.shape}")
    print(f"  meta:   {ROOT / (args.out + '_meta.json')}")


if __name__ == "__main__":
    main()
