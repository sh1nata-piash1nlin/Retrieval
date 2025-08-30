import os
import json
import random
import zlib
from pathlib import Path
from flask import Flask, send_from_directory, request, jsonify, render_template
import numpy as np
import torch
from PIL import Image
from utils.faiss import Faiss 

# Flask app setup
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = BASE_DIR.parent
DATA_DIR = WORKSPACE_DIR / 'data_aichallenge2025'
KEYFRAME_ROOT = DATA_DIR
MEDIA_INFO_DIR = DATA_DIR / 'media-info-aic25-b1' / 'media-info'

app = Flask(__name__)
app.config['DATA_AICHALLENGE2025'] = str(DATA_DIR)

PALETTE = [
    "#0D47A1", "#1A237E", "#283593", "#3F51B5", "#4527A0", "#4A148C",
    "#6A1B9A", "#7B1FA2", "#880E4F", "#AD1457", "#B71C1C", "#BF360C",
    "#E65100", "#5D4037", "#3E2723", "#263238"
]
# Initialize Faiss with all models
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# SIGLIP_FAISS_BIN = str(DATA_DIR / "output_bin" / "faiss_siglip_L2.bin")
SIGLIP_FAISS_BIN = str(DATA_DIR / "output_bin" / "faiss_siglip_L2.bin")
SIGLIP_JSON = str(DATA_DIR / "output_bin" / "keyframes_id_search_siglip2.json")
FDP_FAISS_BIN = str(DATA_DIR / "output_bin" / "faiss_fdp_cosine.bin")
FDP_JSON = str(DATA_DIR / "output_bin" / "keyframes_id_search_fdp.json")
INTERNVIDEO2_FAISS_BIN = str(DATA_DIR / "output_bin" / "faiss_InternVideo2_L2.bin")
INTERNVIDEO2_JSON = str(DATA_DIR / "output_bin" / "keyframes_id_search_internvideo2.json")
BLIP2_FAISS_BIN = str(DATA_DIR / "output_bin" / "faiss_blip2_l2.bin")  # Added BLIP-2 FAISS index
BLIP2_JSON = str(DATA_DIR / "output_bin" / "keyframes_id_search_blip2.json")  # Added BLIP-2 JSON metadata
faiss_index = None
def color_for_video(video_id: str) -> str:
    if not video_id:
        return "#DDDDDD"
    idx = zlib.crc32(video_id.encode("utf-8")) % len(PALETTE)
    return PALETTE[idx]

def get_keyframe_video_dirs():
    if not os.path.exists(KEYFRAME_ROOT):
        return []
    return [
        os.path.join(KEYFRAME_ROOT, d, "keyframes")
        for d in os.listdir(KEYFRAME_ROOT)
        if os.path.isdir(os.path.join(KEYFRAME_ROOT, d))
        and d.startswith("keyframes_Videos_")
        and os.path.isdir(os.path.join(KEYFRAME_ROOT, d, "keyframes"))
    ]

def get_video_folders():
    video_folders = []
    for base_dir in get_keyframe_video_dirs():
        for video_name in sorted(os.listdir(base_dir)):
            video_path = os.path.join(base_dir, video_name)
            if os.path.isdir(video_path):
                base_dir_name = os.path.basename(os.path.dirname(base_dir))
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
            results.append({
                "video_id": vid,
                "frame_num": os.path.splitext(img)[0],
                "image_url": f"/data_aichallenge2025/{base_name}/keyframes/{vid}/{img}",
                "border_color": vid_color,
            })
        if len(results) >= num_images:
            break
    random.shuffle(results)
    return results[:num_images]

# try:
faiss_index = Faiss(
    bin_files=[SIGLIP_FAISS_BIN, FDP_FAISS_BIN, INTERNVIDEO2_FAISS_BIN],
    dict_jsons=[SIGLIP_JSON, FDP_JSON, INTERNVIDEO2_JSON],
    model_types=["siglip2", "fdp", "internvideo2"],  # Specify model types
    device=device
)
# Validate index and JSON alignment
try:
    faiss_index = Faiss(
        bin_files=[SIGLIP_FAISS_BIN, FDP_FAISS_BIN, INTERNVIDEO2_FAISS_BIN, BLIP2_FAISS_BIN],
        dict_jsons=[SIGLIP_JSON, FDP_JSON, INTERNVIDEO2_JSON, BLIP2_JSON],
        model_types=["siglip2", "fdp", "internvideo2", "blip2"],  # Added blip2
        device=device
    )
    # Validate index and JSON alignment
    for idx, (bin_file, json_file, model_type) in enumerate(zip(
            [SIGLIP_FAISS_BIN, FDP_FAISS_BIN, INTERNVIDEO2_FAISS_BIN, BLIP2_FAISS_BIN],
            [SIGLIP_JSON, FDP_JSON, INTERNVIDEO2_JSON, BLIP2_JSON],
            ["siglip2", "fdp", "internvideo2", "blip2"]
        )):
            index_count = faiss_index.indexes[idx].ntotal
            with open(json_file, "r") as f:
                data = json.load(f)
            # Preprocess JSON to match Faiss._read_json
            if model_type == "internvideo2":
                if isinstance(data, list) and all(isinstance(item, list) for item in data):
                    id_to_path = {"paths": data}
                    mapping_count = len(data)  # Number of scenes
                else:
                    raise ValueError(f"Expected nested list for InternVideo2 JSON in {json_file}")
            else:  # For siglip2, fdp, blip2
                if isinstance(data, list) and all(isinstance(item, str) for item in data):
                    id_to_path = {"paths": data}
                    mapping_count = len(data)  # Number of frames
                elif isinstance(data, dict) and "paths" in data:
                    id_to_path = data
                    mapping_count = len(data["paths"])
                else:
                    raise ValueError(f"Expected flat list of strings or dict with 'paths' for {model_type} JSON in {json_file}")
            if index_count != mapping_count:
                raise RuntimeError(
                    f"Index/mapping size mismatch for {bin_file}: index={index_count}, paths={mapping_count}. "
                    "Rebuild the FAISS bin and JSON together."
                )
except Exception as e:
    print(f"Faiss index or mapping not loaded: {e}")
    faiss_index = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/data_aichallenge2025/<path:subpath>")
def serve_data_aichallenge2025(subpath):
    return send_from_directory(app.config['DATA_AICHALLENGE2025'], subpath)

@app.route("/search", methods=["GET"])
def search():
    try:
        results = sample_images_from_videos()
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": f"Random sampling failed: {str(e)}"}), 500

@app.route("/text_search", methods=["POST"])
def text_search():
    query = request.form.get("query")
    top_k = int(request.form.get("top_k", 30))
    model_type = request.form.get("model_type", "siglip2")
    if model_type not in ["siglip2", "fdp", "internvideo2", "blip2"]:
        return jsonify({"error": "Invalid model_type, must be 'siglip2', 'fdp', or 'internvideo2'"}), 400
    if not query or not faiss_index:
        return jsonify({"error": "Missing query or index not loaded"}), 400

    # try:
    results = faiss_index.search_text(query_text=query, top_k=top_k, model_type=model_type)
    # except Exception as e:
    #     return jsonify({"error": f"Search failed: {str(e)}"}), 500

    images = []
    for idx, hits in enumerate(results):
        for hit in hits:
            if model_type == "internvideo2":
                frame_paths = hit["paths"]   # now a list
                if not frame_paths:
                    continue
                # Representative frame (first one)
                rep_img = frame_paths[0]
                parts = rep_img.split("/")
                video_id = parts[-2] if len(parts) >= 2 else ""
                frame_num = parts[-1].split(".")[0] if len(parts) >= 1 else ""
                vid_color = color_for_video(video_id)
                for p in frame_paths:
                    images.append({
                        "image_url": f"/data_aichallenge2025/{p}",   # each frame instead of only rep_img
                        "score": float(hit["score"]),
                        "video_id": video_id,
                        "frame_num": frame_num,
                        "border_color": vid_color,
                        "index_id": idx + 1,
                        "scene_frames": [f"/data_aichallenge2025/{fp}" for fp in frame_paths]  # keep all scene frames
                    })
            else:
                # Handle single-frame results
                img_path = hit["path"]
                if img_path == "unknown":
                    continue
                parts = img_path.split("/")
                video_id = parts[-2] if len(parts) >= 2 else ""
                frame_num = parts[-1].split(".")[0] if len(parts) >= 1 else ""
                vid_color = color_for_video(video_id)
                images.append({
                    "image_url": f"/data_aichallenge2025/{img_path}",
                    "score": float(hit["score"]),
                    "video_id": video_id,
                    "frame_num": frame_num,
                    "border_color": vid_color,
                    "index_id": idx + 1,
                })
    # print(images)
    return jsonify({"results": images})


@app.route("/image_search", methods=["POST"])
def image_search():
    if 'image' not in request.files or not faiss_index:
        return jsonify({"error": "Missing image or index not loaded"}), 400
    image_file = request.files['image']
    top_k = int(request.form.get("top_k", 30))
    model_type = request.form.get("model_type", "siglip2")
    if model_type not in ["siglip2", "fdp", "internvideo2", "blip2"]:
        return jsonify({"error": "Invalid model_type, must be 'siglip2', 'fdp', or 'internvideo2'"}), 400

    try:
        image = Image.open(image_file).convert("RGB")
        results = faiss_index.search_image(query_image=image, top_k=top_k, model_type=model_type)
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

    images = []
    for idx, hits in enumerate(results):
        for hit in hits:
            img_path = hit["path"]
            if img_path == "unknown":
                continue
            parts = img_path.split("/")
            video_id = parts[-2] if len(parts) >= 2 else ""
            frame_num = parts[-1].split(".")[0] if len(parts) >= 1 else ""
            vid_color = color_for_video(video_id)
            images.append({
                "image_url": f"/data_aichallenge2025/{img_path}",
                "score": float(hit["score"]),
                "video_id": video_id,
                "frame_num": frame_num,
                "border_color": vid_color,
                "index_id": idx + 1,
            })

    return jsonify({"results": images})

@app.route("/neighboring_frames", methods=["POST"])
def neighboring_frames():
    video_id = request.form.get("video_id")
    frame_num = request.form.get("frame_num")
    model_type = request.form.get("model_type", "siglip2")

    # Validate inputs
    if not video_id or not frame_num:
        print(f"Error: Missing video_id ({video_id}) or frame_num ({frame_num})")
        return jsonify({"error": "Missing video_id or frame_num"}), 400
    try:
        frame_num = int(frame_num)
    except ValueError:
        print(f"Error: Invalid frame_num ({frame_num}), must be an integer")
        return jsonify({"error": "Invalid frame_num, must be an integer"}), 400
    if model_type not in ["siglip2", "fdp", "internvideo2", "blip2"]:
        print(f"Error: Invalid model_type ({model_type})")
        return jsonify({"error": "Invalid model_type, must be 'siglip2', 'fdp', 'internvideo2', or 'blip2'"}), 400

    try:
        # Load the merged JSON metadata
        json_path = os.path.join(DATA_DIR, "output_bin", "keyframes_id_search_asr.json")
        metadata = []
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            print(f"Loaded metadata with {len(metadata)} segments")
        else:
            print(f"Warning: Metadata JSON file not found at {json_path}, proceeding without text")

        # Find the keyframe directory for the video
        keyframe_dirs = get_keyframe_video_dirs()
        video_path = None
        base_dir_name = None
        for base_dir in keyframe_dirs:
            potential_path = os.path.join(base_dir, video_id)
            if os.path.isdir(potential_path):
                video_path = potential_path
                base_dir_name = os.path.basename(os.path.dirname(base_dir))
                break

        if not video_path:
            print(f"Error: Video directory not found for {video_id}")
            return jsonify({"error": f"Video {video_id} not found"}), 404

        # Get all frame files in the video directory
        frame_files = sorted(
            [f for f in os.listdir(video_path) if f.lower().endswith(".jpg")],
            key=lambda x: int(os.path.splitext(x)[0])
        )
        if not frame_files:
            print(f"Error: No frames found in {video_path}")
            return jsonify({"error": f"No frames found for video {video_id}"}), 404

        print(f"Found {len(frame_files)} frame files: {[f for f in frame_files[:5]]}...")

        # Find the current frame or the closest frame
        frame_nums = [int(os.path.splitext(f)[0]) for f in frame_files]
        current_frame = f"{frame_num:06d}.jpg"
        if current_frame not in frame_files:
            closest_frame_num = min(frame_nums, key=lambda x: abs(x - frame_num))
            current_frame = f"{closest_frame_num:06d}.jpg"
            frame_num = closest_frame_num
            print(f"Warning: Frame {frame_num:06d}.jpg not found, using closest frame {current_frame}")

        current_idx = frame_files.index(current_frame)
        frames = []

        # Helper function to get text from metadata
        def get_text_for_frame(frame_num):
            for segment in metadata:
                if (segment["video"] == f"{video_id}.mp4" and
                    segment["start_frame"] <= frame_num <= segment["end_frame"]):
                    return segment["text"]
            return "No script available"

        # Add previous frame (if exists)
        if current_idx > 0:
            prev_frame = frame_files[current_idx - 1]
            prev_frame_num = int(os.path.splitext(prev_frame)[0])
            frames.append({
                "image_url": f"/data_aichallenge2025/{base_dir_name}/keyframes/{video_id}/{prev_frame}",
                "frame_num": prev_frame_num,
                "video_id": video_id,
                "text": get_text_for_frame(prev_frame_num)
            })
            print(f"Added previous frame: {prev_frame}")

        # Add current frame
        frames.append({
            "image_url": f"/data_aichallenge2025/{base_dir_name}/keyframes/{video_id}/{current_frame}",
            "frame_num": frame_num,
            "video_id": video_id,
            "text": get_text_for_frame(frame_num)
        })
        print(f"Added current frame: {current_frame}")

        # Add next frame (if exists)
        if current_idx < len(frame_files) - 1:
            next_frame = frame_files[current_idx + 1]
            next_frame_num = int(os.path.splitext(next_frame)[0])
            frames.append({
                "image_url": f"/data_aichallenge2025/{base_dir_name}/keyframes/{video_id}/{next_frame}",
                "frame_num": next_frame_num,
                "video_id": video_id,
                "text": get_text_for_frame(next_frame_num)
            })
            print(f"Added next frame: {next_frame}")

        # Sort frames by frame_num to ensure correct order
        frames.sort(key=lambda x: x["frame_num"])

        if not frames:
            print(f"Error: No valid frames found for video {video_id}")
            return jsonify({"error": "No valid frames found"}), 404

        print(f"Returning {len(frames)} frames: {[f['frame_num'] for f in frames]}")
        return jsonify({"frames": frames})

    except Exception as e:
        print(f"Error: Failed to fetch neighboring frames: {str(e)}")
        return jsonify({"error": f"Failed to fetch neighboring frames: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888, debug=False, threaded=True)