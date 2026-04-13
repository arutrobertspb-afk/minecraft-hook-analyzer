#!/usr/bin/env python3
"""
Pixel Rage — Hook Analyzer POC
Takes YouTube Shorts URLs, extracts first frame, runs CLIP,
scores similarity to emotion anchors.
"""
import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import yt_dlp
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Emotion anchors — text prompts that represent different hook types
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
]

OUT_DIR = Path.home() / "pixel_rage_research" / "frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_metadata_and_video(url: str, idx: int):
    """Download video (lowest quality ok, we only need frame 0) + metadata."""
    out_template = str(OUT_DIR / f"vid_{idx}.%(ext)s")
    ydl_opts = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"player_client": ["ios", "android", "web"]}},
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    # Find downloaded file
    video_path = None
    for ext in ("mp4", "webm", "mkv"):
        p = OUT_DIR / f"vid_{idx}.{ext}"
        if p.exists():
            video_path = p
            break
    return video_path, {
        "id": info.get("id"),
        "title": info.get("title"),
        "views": info.get("view_count"),
        "likes": info.get("like_count"),
        "duration": info.get("duration"),
        "channel": info.get("channel"),
        "channel_followers": info.get("channel_follower_count"),
        "upload_date": info.get("upload_date"),
    }


def extract_first_frame(video_path: Path, idx: int) -> Path:
    """Extract frame at t=0.1s (slightly past 0 to avoid black intro)."""
    frame_path = OUT_DIR / f"frame_{idx}.jpg"
    cmd = [
        FFMPEG, "-y", "-ss", "0.1", "-i", str(video_path),
        "-vframes", "1", "-q:v", "2", str(frame_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return frame_path


def main(urls):
    print(f"Loading CLIP model (first run downloads ~600MB)...", flush=True)
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Encode text anchors once
    with torch.no_grad():
        text_in = processor(text=EMOTION_ANCHORS, return_tensors="pt", padding=True)
        text_feat = model.get_text_features(**text_in)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    results = []
    for i, url in enumerate(urls):
        print(f"\n[{i+1}/{len(urls)}] {url}", flush=True)
        try:
            vpath, meta = fetch_metadata_and_video(url, i)
            print(f"  {meta['title'][:70]}", flush=True)
            v = meta['views'] or 0
            lk = meta['likes'] or 0
            print(f"  views={v:,} likes={lk:,} "
                  f"ch={meta['channel']} subs={meta.get('channel_followers')}", flush=True)

            frame = extract_first_frame(vpath, i)
            img = Image.open(frame).convert("RGB")

            with torch.no_grad():
                img_in = processor(images=img, return_tensors="pt")
                img_feat = model.get_image_features(**img_in)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sims = (img_feat @ text_feat.T).squeeze(0).tolist()

            scored = sorted(zip(EMOTION_ANCHORS, sims), key=lambda x: -x[1])
            meta["top_emotions"] = [(e, round(s, 4)) for e, s in scored[:5]]
            meta["frame_path"] = str(frame)
            results.append(meta)

            print("  top emotions:")
            for emo, sc in scored[:5]:
                print(f"    {sc:.4f}  {emo}")
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            results.append({"url": url, "error": str(e)})

    # Save JSON
    out_json = OUT_DIR.parent / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_json}")

    # Compact summary
    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            continue
        views = r["views"] or 0
        subs = r.get("channel_followers") or 0
        norm = views / max(subs, 1)
        top = r["top_emotions"][0]
        print(f"views={views:>12,}  views/sub={norm:6.2f}  "
              f"top={top[1]:.3f} {top[0][:40]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_hooks.py <url1> <url2> ...")
        sys.exit(1)
    main(sys.argv[1:])
