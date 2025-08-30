import torch
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
import json
import sys
import glob
import argparse
from blip2 import BLIP2Model
# -------------------------
# Embedding + FAISS utils
# -------------------------
def add_vector(embedding, index):
    index.add(embedding)

# -------------------------
# Create .npy files for each video
# -------------------------
def create_npy(input_data_path: str):
    """Create .npy files for each video's keyframes in .../keyframes_Videos_L*/blip2_npy/<video_id>/feats.npy."""
    # Ensure input path exists
    if not os.path.exists(input_data_path):
        print(f"ERROR: {input_data_path} does not exist")
        return

    # Initialize BLIP-2 model
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    blip2_model = BLIP2Model(device)

    # Collect all keyframe directories
    root_path = Path(input_data_path)
    video_folders = []
    for subdir in sorted(os.listdir(root_path)):
        if "keyframes_Videos_L" not in subdir:
            continue
        keyframes_path = os.path.join(root_path, subdir, "keyframes")
        if not os.path.isdir(keyframes_path):
            continue
        for video_dir in sorted(os.listdir(keyframes_path)):
            video_path = os.path.join(keyframes_path, video_dir)
            if os.path.isdir(video_path):
                video_folders.append((subdir, video_path))

    # Process each video folder
    for subdir, video_path in tqdm(video_folders, desc="Processing videos"):
        video_name = os.path.basename(video_path)  # e.g., L21_V001
        images_path_list = sorted(
            glob.glob(os.path.join(video_path, "*.jpg")),
            key=lambda x: x.split('/')[-1].replace('.jpg', '')
        )
        if not images_path_list:
            print(f"Warning: No images found in {video_path}, skipping")
            continue

        # Generate embeddings for all keyframes in the video
        re_feats = []
        video_frame_paths = []
        for img_path in images_path_list:
            # print("DEBUG: img_path type =", type(img_path), "value =", img_path)
            # if not isinstance(img_path, str):
            #     print(f"⚠️ Skipping invalid path: {img_path}")
            #     continue

            try:
                # image = Image.open(img_path).convert("RGB")
                emb = blip2_model.image_encoder(img_path)
                re_feats.append(emb)
                relative_path = os.path.relpath(img_path, start=root_path)
                formatted_path = relative_path.replace(os.path.sep, '/')
                video_frame_paths.append(formatted_path)
            except Exception as e:
                print(f"⚠️ Skipped {img_path}: {e}")


        # Save factored-out embeddings to .npy file in .../keyframes_Videos_L*/blip2_npy/<video_id>/
        if re_feats:
            npy_dir = os.path.join(root_path, subdir, "blip2_npy", video_name)
            os.makedirs(npy_dir, exist_ok=True)
            npy_path = os.path.join(npy_dir, "feats.npy")
            np.save(npy_path, np.array(re_feats))
            print(f"Saved embeddings to {npy_path}")

            # Save frame paths to JSON for mapping
            json_path = os.path.join(npy_dir, "frame_paths.json")
            with open(json_path, "w") as f:
                json.dump(video_frame_paths, f, indent=4)
            print(f"Saved frame paths to {json_path}")

# -------------------------
# Create FAISS index from .npy files
# -------------------------
def create_bin(input_npy_path: str, output_bin_path: str, method="L2", feature_shape=768):
    """Create FAISS index from .npy files."""
    if method == "L2":
        index = faiss.IndexFlatL2(feature_shape)
    elif method == "cosine":
        index = faiss.IndexFlatIP(feature_shape)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    else:
        raise ValueError(f"{method} not supported")

    # Collect all .npy files
    npy_files = glob.glob(os.path.join(input_npy_path, "**/blip2_npy/**/feats.npy"), recursive=True)
    if not npy_files:
        raise ValueError("No .npy file found, check input path!")

    # Store mapping of indices to frame paths
    id_to_path = []
    global_idx = 0

    for npy_file in tqdm(npy_files, desc="Building FAISS index"):
        # Load embeddings
        feats = np.load(npy_file).astype(np.float32)
        if method == "cosine":
            faiss.normalize_L2(feats)

        # Add to FAISS index
        index.add(feats)

        # Load corresponding frame paths
        json_file = os.path.join(os.path.dirname(npy_file), "frame_paths.json")
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping path mapping")
            continue
        with open(json_file, "r") as f:
            frame_paths = json.load(f)
        if len(frame_paths) != feats.shape[0]:
            print(f"Warning: Mismatch between embeddings ({feats.shape[0]}) and paths ({len(frame_paths)}) in {npy_file}")
            continue
        id_to_path.extend(frame_paths)

    # Save FAISS index
    os.makedirs(output_bin_path, exist_ok=True)
    out_path = os.path.join(output_bin_path, f"faiss_blip2_{method}.bin")
    faiss.write_index(index, out_path)
    print(f"Saved FAISS index to {out_path}")

    # Save id_to_path to JSON
    json_path = os.path.join(output_bin_path, "keyframes_id_search_blip2.json")
    with open(json_path, "w") as f:
        json.dump(id_to_path, f, indent=4)
    print(f"Saved id_to_path to {json_path}")

# -------------------------
# Main function
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img2npy", action='store_true', help="Convert images to .npy files")
    parser.add_argument("--npy2bin", action='store_true', help="Convert .npy files to FAISS .bin file")
    parser.add_argument('-i', required=True, type=str, help="Input keyframes dir (for img2npy) or npy dir (for npy2bin)")
    parser.add_argument('-o', required=True, type=str, help="Output bin dir (for npy2bin; ignored for img2npy)")
    parser.add_argument('--method', type=str, default="L2", choices=["L2", "cosine"], help="FAISS index method")
    args = parser.parse_args()

    if args.img2npy:
        create_npy(args.i)
    if args.npy2bin:
        create_bin(args.i, args.o, method=args.method)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    main()