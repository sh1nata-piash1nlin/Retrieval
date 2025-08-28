import os
import json
import random
import zlib
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from langdetect import detect
import numpy as np
import torch

from utils.faiss import Faiss
from models.SigLip2 import embed_siglip, processor, model, embed_siglip_text
from utils.query_processing import Translation

BASE_DIR = Path(__file__).resolve().parent
KEYFRAME_ROOT = BASE_DIR / "static" / "keyframe"  
app = Flask(__name__)

PALETTE = [
    "#0D47A1",  # deep blue
    "#1A237E",  # indigo 900
    "#283593",  # indigo 800
    "#3F51B5",  # indigo 500
    "#4527A0",  # deep purple
    "#4A148C",  # purple 900
    "#6A1B9A",  # purple 800
    "#7B1FA2",  # purple 700
    "#880E4F",  # dark magenta
    "#AD1457",  # crimson/magenta
    "#B71C1C",  # dark red
    "#BF360C",  # burnt orange
    "#E65100",  # dark orange
    "#5D4037",  # brown
    "#3E2723",  # very dark brown
    "#263238",  # charcoal blue-grey
]

def color_for_video(video_id: str) -> str:
    """Deterministically pick a color from PALETTE for a given video_id."""
    if not video_id:
        return "#DDDDDD"
    idx = zlib.crc32(video_id.encode("utf-8")) % len(PALETTE)
    return PALETTE[idx]

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

@app.route("/search", methods=["POST"])
def search():
    results = sample_images_from_videos()
    return render_template("_search_results.html", results=results)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=2714, debug=True)
