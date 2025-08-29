import os
import json
import random
import zlib
from pathlib import Path

from flask import Flask, render_template, request, jsonify, url_for
from langdetect import detect
import numpy as np
import torch

from utils.faiss import Faiss
from models.SigLip2 import embed_siglip, processor, model, embed_siglip_text
from utils.query_processing import Translation

BASE_DIR = Path(__file__).resolve().parent
KEYFRAME_ROOT = BASE_DIR / "static" / "keyframe"  
app = Flask(__name__)


def get_keyframe_video_dirs():
    keyframe_root = os.path.join(app.static_folder, "keyframe")
    if not os.path.exists(keyframe_root):
        return []
    return [
        os.path.join(keyframe_root, d)
        for d in os.listdir(keyframe_root)
        if os.path.isdir(os.path.join(keyframe_root, d))
        and d.startswith("keyframes_Videos_")
    ]

def get_video_folders():
    video_folders = []
    for base_dir in get_keyframe_video_dirs():
        # Each base_dir is like /static/keyframe/keyframes_Videos_L21
        for video_name in sorted(os.listdir(base_dir)):
            video_path = os.path.join(base_dir, video_name)
            if os.path.isdir(video_path):
                base_dir_name = os.path.basename(base_dir)
                video_folders.append((video_name, video_path, base_dir_name))
    return video_folders

def sample_images_from_videos(num_images=30, images_per_video=(3, 4)):
    video_folders = get_video_folders()
    random.shuffle(video_folders)
    results = []

    for vid, path, base_name in video_folders:
        images = [f for f in os.listdir(path) if f.lower().endswith(".jpg")]
        if not images:
            continue

        k = random.randint(*images_per_video)
        chosen = random.sample(images, min(k, len(images)))
        vid_color = color_for_video(vid)

        for img in chosen:
            results.append(
                {
                    "video_id": vid,
                    "frame_num": os.path.splitext(img)[0],
                    "image_url": f"/static/keyframe/{base_name}/{vid}/{img}",
                    "border_color": vid_color,
                }
            )
        if len(results) >= num_images:
            break

    random.shuffle(results)  # mix videos
    return results[:num_images]

def find_video_dir(video_id: str):
    for base_dir in get_keyframe_video_dirs():
        direct = os.path.join(base_dir, video_id)
        nested = os.path.join(base_dir, "keyframes", video_id)

        if os.path.isdir(direct):
            return direct, os.path.relpath(direct, app.static_folder)

        if os.path.isdir(nested):
            return nested, os.path.relpath(nested, app.static_folder)

    return None, None

def _hsl_to_hex(h: int, s: int, l: int) -> str:
    """Convert HSL (0–360, 0–100, 0–100) to #RRGGBB."""
    s /= 100.0
    l /= 100.0
    c = (1 - abs(2*l - 1)) * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = l - c / 2.0

    if   0 <= h < 60:   r, g, b = c, x, 0
    elif 60 <= h < 120: r, g, b = x, c, 0
    elif 120 <= h < 180:r, g, b = 0, c, x
    elif 180 <= h < 240:r, g, b = 0, x, c
    elif 240 <= h < 300:r, g, b = x, 0, c
    else:               r, g, b = c, 0, x

    r = int(round((r + m) * 255))
    g = int(round((g + m) * 255))
    b = int(round((b + m) * 255))
    return f"#{r:02X}{g:02X}{b:02X}"

def color_for_video(video_id: str) -> str:
    """
    Deterministic vivid color for each video_id.
    - Avoids the green hue band (used by selection UI).
    - Adds slight lightness variation to reduce collisions further.
    """
    if not video_id:
        return "#888888"

    crc = zlib.crc32(video_id.encode("utf-8"))
    h = crc % 360

    # Avoid selection-green band (~120°). Shift it away.
    if 100 <= h <= 150:
        h = (h + 60) % 360

    # Vibrant but still readable on white; slight variation per ID
    s = 72                              # saturation
    l = 42 + ((crc >> 12) % 10)         # 42..51 (darker/lighter steps)

    return _hsl_to_hex(h, s, l)

@app.route("/")
def home():
    return render_template("home.html")

SIGLIP_FAISS_BIN = str(BASE_DIR / "faiss_siglip_L2.bin")
SIGLIP_JSON = str(BASE_DIR / "keyframes_id_search_siglip2.json")
siglip_index = None
siglip_id_to_path = None

try:
    siglip_index = Faiss(SIGLIP_FAISS_BIN)
    with open(SIGLIP_JSON, "r") as f:
        siglip_id_to_path = json.load(f)

    index_count = siglip_index.index.ntotal  # sanity check
    mapping_count = len(siglip_id_to_path)
    if index_count != mapping_count:
        raise RuntimeError(
            f"SigLIP index/mapping size mismatch: index={index_count}, paths={mapping_count}. "
            "Rebuild the FAISS bin and JSON together."
        )

    translation = Translation(from_lang="vi", to_lang="en", mode="googletrans")

except Exception as e:
    print(f"SigLIP2 index or mapping not loaded: {e}")
    siglip_index = None
    siglip_id_to_path = None

@app.route("/siglip2_search", methods=["POST"])
def siglip2_search():
    query = request.form.get("query")
    top_k = int(request.form.get("top_k", 30))
    if not query or not siglip_index or not siglip_id_to_path:
        return jsonify({"error": "Missing query or index"}), 400

    try:
        lang = Translation()(query) if detect(query) == "vi" else query
    except Exception:
        lang = query

    # Build query embedding
    _ = processor(text=lang, return_tensors="pt").to(model.device)  # keep processor device usage consistent
    with torch.no_grad():
        query_vec = embed_siglip_text(lang)

    # Search
    results = siglip_index.search(query_vecs=query_vec, top_k=top_k)[0]

    # Build response
    images = []
    for hit in results:
        idx = hit["id"]
        score = hit["score"]
        img_path = siglip_id_to_path[idx]

        parts = img_path.split("/")
        video_id = parts[-2] if len(parts) >= 2 else ""
        frame_num = parts[-1].split(".")[0] if len(parts) >= 1 else ""
        vid_color = color_for_video(video_id)

        images.append(
            {
                "image_url": f"/static/keyframe/{img_path}",
                "score": score,
                "video_id": video_id,
                "frame_num": frame_num,
                "border_color": vid_color,  # pastel color for this video
            }
        )

    return render_template("_search_results.html", results=images)


@app.route('/frame_window')
def frame_window():
    video_id = request.args.get("video_id")
    frame_str = request.args.get("frame")
    window = int(request.args.get("w", 50))
    if not video_id or not frame_str:
        return "Missing params", 400

    abs_dir, rel_dir = find_video_dir(video_id)   # rel_dir is path *from* app.static_folder
    if not abs_dir:
        return f"Video {video_id} not found", 404

    files = [f for f in os.listdir(abs_dir) if f.lower().endswith(".jpg")]
    if not files:
        return "No frames found", 404

    files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # numeric sort

    target_name = f"{frame_str}.jpg"
    try:
        idx = files.index(target_name)
    except ValueError:
        target_num = int(frame_str)
        idx = min(range(len(files)), key=lambda i: abs(int(os.path.splitext(files[i])[0]) - target_num))

    start = max(0, idx - window)
    end = min(len(files), idx + window + 1)
    subset = files[start:end]

    # IMPORTANT: convert rel_dir to web/posix path OR just use url_for
    web_rel = rel_dir.replace(os.path.sep, "/")  # safe even on Linux/Mac

    frames = []
    for fn in subset:
        num = os.path.splitext(fn)[0]
        # Prefer url_for so Flask handles STATIC_URL_PATH etc.
        url = url_for("static", filename=f"{web_rel}/{fn}")
        frames.append({"image_url": url, "frame_num": num, "is_current": (fn == target_name)})

    return render_template("_frame_window.html",
                           video_id=video_id,
                           current_frame=frame_str,
                           frames=frames)

@app.route("/search", methods=["POST"])
def search():
    results = sample_images_from_videos()
    return render_template("_search_results.html", results=results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=2714, debug=True)
