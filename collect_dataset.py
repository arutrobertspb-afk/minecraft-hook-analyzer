#!/usr/bin/env python3
"""
Collect dataset: search YouTube Shorts, download, extract first 5 seconds
(frames at 0.5, 1.5, 2.5, 3.5, 4.5s), compute CLIP embeddings.

Saves:
  {out}_embeds.npy       — shape (N, 512)    mean embedding per video
  {out}_frame_embeds.npy — shape (N, 5, 512) per-frame embeddings (timeline)
  {out}_meta.json        — metadata list

Usage:
  python3 collect_dataset.py --n 100
  python3 collect_dataset.py --n 50 --out run1
  python3 collect_dataset.py --n 50 --append  # add to existing dataset
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

FRAME_TIMES = [0.5, 1.5, 2.5, 3.5, 4.5]  # seconds into video


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


def extract_frames(video_path: Path, idx: int, duration: float = 60.0) -> list:
    """Extract 5 frames at FRAME_TIMES seconds. Returns list of Paths (only existing ones)."""
    frame_paths = []
    for ti, t in enumerate(FRAME_TIMES):
        if t >= duration:
            break
        frame_path = FRAMES_DIR / f"frame_{idx}_t{ti}.jpg"
        cmd = [
            FFMPEG, "-y", "-ss", str(t), "-i", str(video_path),
            "-vframes", "1", "-q:v", "2", str(frame_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if frame_path.exists():
                frame_paths.append(frame_path)
        except Exception:
            pass
    return frame_paths


KEYWORD_RE = re.compile(r"minecraft|steve|creeper|enderman|mcyt", re.IGNORECASE)


def embed_frames(frame_paths: list, model, processor) -> np.ndarray:
    """Embed a list of frames, return array of shape (len, 512)."""
    vecs = []
    for fp in frame_paths:
        img = Image.open(fp).convert("RGB")
        with torch.no_grad():
            img_in = processor(images=img, return_tensors="pt")
            feat = model.get_image_features(**img_in)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            vecs.append(feat.squeeze(0).cpu().numpy().astype(np.float32))
    return np.stack(vecs, axis=0)  # (K, 512)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="Target number of videos to collect")
    ap.add_argument("--sources", type=str, nargs="+",
                    default=[
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
                        "minecraft shorts",
                        "minecraft shorts viral",
                        "minecraft shorts funny",
                        "minecraft creeper shorts",
                        "minecraft steve shorts",
                    ],
                    help="Sources: hashtag (#tag), search query, or full URL")
    ap.add_argument("--per-source", type=int, default=40, help="Candidates per source")
    ap.add_argument("--out", type=str, default="dataset", help="Output basename (no ext)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing dataset instead of overwriting")
    ap.add_argument("--keyword-filter", action="store_true", default=True,
                    help="Require 'minecraft' in title/channel (default on)")
    ap.add_argument("--no-keyword-filter", dest="keyword_filter", action="store_false")
    args = ap.parse_args()

    # Load existing data if appending
    existing_ids = set()
    existing_embeds = []
    existing_frame_embeds = []
    existing_metas = []
    if args.append:
        embeds_path = ROOT / f"{args.out}_embeds.npy"
        frame_embeds_path = ROOT / f"{args.out}_frame_embeds.npy"
        meta_path = ROOT / f"{args.out}_meta.json"
        if embeds_path.exists() and meta_path.exists():
            existing_embeds = list(np.load(embeds_path))
            existing_metas = json.loads(meta_path.read_text())
            existing_ids = {m.get("id") for m in existing_metas if m.get("id")}
            if frame_embeds_path.exists():
                existing_frame_embeds = list(np.load(frame_embeds_path))
            print(f"Appending to existing {len(existing_metas)} videos", flush=True)

    print(f"Collecting from {len(args.sources)} sources, {args.per_source} each", flush=True)
    entries = search_shorts(args.sources, args.per_source)
    print(f"Got {len(entries)} unique candidates after dedupe", flush=True)

    print("Loading CLIP...", flush=True)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    embeds = []         # mean embedding per video, shape (512,)
    frame_embeds = []   # per-frame embeddings, shape (5, 512)
    metas = []
    skipped = {"filter": 0, "keyword": 0, "download": 0, "frame": 0, "duplicate": 0}
    start_idx = len(existing_metas)

    for i, e in enumerate(entries):
        if len(metas) >= args.n:
            break
        vid = e.get("id")
        if not vid:
            continue

        # Skip duplicates when appending
        if vid in existing_ids:
            skipped["duplicate"] += 1
            continue

        print(f"[{len(metas)+1}/{args.n}] {vid} (tried {i+1})", flush=True)

        vpath, meta = fetch_video(vid, start_idx + len(metas))
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

        # Extract 5 frames from first 5 seconds
        duration = meta.get("duration") or 60.0
        try:
            fpaths = extract_frames(vpath, start_idx + len(metas), duration)
        except Exception as ex:
            print(f"  frame extract failed: {ex}", flush=True)
            skipped["frame"] += 1
            continue

        if not fpaths:
            print("  no frames extracted", flush=True)
            skipped["frame"] += 1
            continue

        # Embed all available frames
        try:
            frame_vecs = embed_frames(fpaths, model, processor)  # (K, 512)
        except Exception as ex:
            print(f"  embed failed: {ex}", flush=True)
            skipped["frame"] += 1
            continue

        # Pad to 5 frames by repeating last if video is shorter than 4.5s
        K = frame_vecs.shape[0]
        if K < len(FRAME_TIMES):
            pad = np.tile(frame_vecs[-1:], (len(FRAME_TIMES) - K, 1))
            frame_vecs_padded = np.concatenate([frame_vecs, pad], axis=0)
        else:
            frame_vecs_padded = frame_vecs

        # Mean embedding for the main embeds array
        mean_vec = frame_vecs.mean(axis=0)
        mean_vec = (mean_vec / (np.linalg.norm(mean_vec) + 1e-9)).astype(np.float32)

        meta["frame_path"] = str(fpaths[0])   # first frame (backward compat)
        meta["frame_paths"] = [str(p) for p in fpaths]
        meta["n_frames"] = K

        embeds.append(mean_vec)
        frame_embeds.append(frame_vecs_padded)
        metas.append(meta)

        print(f"  OK  {K} frames  views={meta['views']:,} subs={meta['channel_subs']:,} "
              f"{meta['width']}x{meta['height']} | {meta['title'][:55]}", flush=True)

        # Delete video file to save disk
        try:
            vpath.unlink()
        except Exception:
            pass

    if not embeds and not existing_embeds:
        print("No videos collected.")
        sys.exit(1)

    # Merge with existing if appending
    all_embeds = existing_embeds + embeds
    all_frame_embeds = existing_frame_embeds + frame_embeds
    all_metas = existing_metas + metas

    # Save
    embeds_arr = np.stack(all_embeds, axis=0)
    frame_embeds_arr = np.stack(all_frame_embeds, axis=0)

    np.save(ROOT / f"{args.out}_embeds.npy", embeds_arr)
    np.save(ROOT / f"{args.out}_frame_embeds.npy", frame_embeds_arr)
    with open(ROOT / f"{args.out}_meta.json", "w") as f:
        json.dump(all_metas, f, indent=2, ensure_ascii=False)

    new_count = len(metas)
    total_count = len(all_metas)
    print(f"\nCollected {new_count} new videos (total: {total_count})")
    print(f"  skipped: {skipped}")
    print(f"  embeds:       {ROOT / (args.out + '_embeds.npy')}  shape={embeds_arr.shape}")
    print(f"  frame_embeds: {ROOT / (args.out + '_frame_embeds.npy')}  shape={frame_embeds_arr.shape}")
    print(f"  meta:         {ROOT / (args.out + '_meta.json')}")


if __name__ == "__main__":
    main()
