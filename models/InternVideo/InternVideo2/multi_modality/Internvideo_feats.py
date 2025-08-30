import numpy as np
import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import glob
import faiss
import argparse
import sys
import json
import re

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from multi_modality.utils.config_impl import Config, eval_dict_leaf
from multi_modality.utils.utils_impl import setup_internvideo2

# Normalization constants
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(data):
    return (data / 255.0 - v_mean) / v_std

def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    """Convert a list of frames to a tensor for the model."""
    if len(vid_list) < fnum:
        vid_list = vid_list + [vid_list[-1]] * (fnum - len(vid_list))
    step = max(1, len(vid_list) // fnum)
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    return torch.from_numpy(vid_tube).to(device, non_blocking=True).float()

def read_txt(txt_path):
    """Read scene boundaries from a text file."""
    ranges_list = []
    with open(txt_path, 'r') as file:
        for line in file:
            stripped_line = line.strip().strip('[]')
            numbers = list(map(int, stripped_line.split()))
            ranges_list.append(numbers)
    return ranges_list

def get_scene_groups(segment_index_list, group_size=3, stride=1):
    """Generate groups of consecutive scenes with a given stride."""
    return [segment_index_list[i:i + group_size]
            for i in range(0, len(segment_index_list) - group_size + 1, stride)]

def segment_images_by_segment_index(images_path_list, segment_index_list):
    """Group images by segment indices, handling arbitrary keyframe names."""
    # Extract numerical index from filename (e.g., 'frame_53.jpg' -> 53)
    def extract_index(filename):
        # Match any number in the filename (e.g., 'frame_53.jpg' or '00053.jpg')
        match = re.search(r'\d+', filename.split('/')[-1].replace('.jpg', ''))
        return int(match.group()) if match else 0

    images_index_array = np.array([extract_index(x) for x in images_path_list])
    images_path_list = np.array(images_path_list)
    segment_images_path_list, segment_frame_list = [], []
    for segment_index in segment_index_list:
        mask = (images_index_array >= segment_index[0]) & (images_index_array <= segment_index[1])
        segment_images_path = images_path_list[mask]
        segment_images_path_list.append(segment_images_path)
        frames = [cv2.imread(str(img_path)) for img_path in segment_images_path if cv2.imread(str(img_path)) is not None]
        segment_frame_list.append(frames)
    return segment_images_path_list, segment_frame_list


class InternVideo2Feats:
    def __init__(self, device: str, active_model=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if active_model:
            self.config = Config.from_file("/workspace/huy_aichallenge/models/InternVideo/InternVideo2/multi_modality/internvideo2_stage2_config.py")
            config = eval_dict_leaf(self.config)
            model_pth = "/workspace/huy_aichallenge/models/InternVideo/InternVideo2/multi_modality/weights/InternVideo2-stage2_1b-224p-f4.pt"
            if not os.path.exists(model_pth):
                raise FileNotFoundError(f"Model checkpoint {model_pth} not found")
            config['pretrained_path'] = model_pth
            config['device'] = str(self.device)  # Update config device
            intern_model, self.tokenizer = setup_internvideo2(config)
            self.vlm = intern_model  # Device transfer handled in setup_internvideo2
            print("Img2npy mode ----- Load model successfully")
        else:
            print("Npy2bin mode")

    def create_npy(self, input_data_path: str, output_npy_dir: str, group_size=3, stride=1):
        """Create .npy files for groups of scenes in .../keyframes_Videos_L*/siglip_npy/<video_id>/feats.npy."""
        fn = self.config.get('num_frames', 8)
        size_t = self.config.get('size_t', 224)

        if not os.path.exists(input_data_path):
            print(f"ERROR: {input_data_path} does not exist")
            return

        # Collect all keyframe directories
        root_path = Path(input_data_path)
        video_folders = []
        for subdir in sorted(os.listdir(root_path)):
            if "keyframes_Videos_K" not in subdir:
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
            folder_path = video_path
            txt_path = os.path.join(root_path, subdir, 'keyframes', f"{video_name}.txt")

            # Get all keyframe images
            images_path = glob.glob(os.path.join(folder_path, "*.jpg"))
            images_path_list = sorted(images_path, key=lambda x: int(re.search(r'\d+', x.split('/')[-1].replace('.jpg', '')).group()))
            if not images_path_list:
                print(f"Warning: No images found in {folder_path}, skipping")
                continue

            # Read frames
            frames = [cv2.imread(str(img_path)) for img_path in images_path_list if cv2.imread(str(img_path)) is not None]
            if not frames:
                print(f"Warning: No valid frames found in {folder_path}, skipping")
                continue

            re_feats = []
            group_frame_paths = []

            if os.path.exists(txt_path):
                # Case 1: Use scene boundaries from txt file
                print(f"Found {txt_path}, processing with scene boundaries")
                segment_index_list = read_txt(txt_path)
                scene_groups = get_scene_groups(segment_index_list, group_size=group_size, stride=stride)
                segment_images_path_list, segment_frame_list = segment_images_by_segment_index(images_path_list, segment_index_list)

                for group_idx, scene_group in enumerate(scene_groups):
                    group_frames = []
                    group_paths = []
                    for scene_idx in scene_group:
                        scene_idx_pos = segment_index_list.index(scene_idx)
                        scene_frames = segment_frame_list[scene_idx_pos]
                        scene_paths = segment_images_path_list[scene_idx_pos]
                        group_frames.extend(scene_frames)
                        group_paths.extend(scene_paths)

                    if len(group_frames) > 0:
                        step = max(1, len(group_frames) // fn)
                        sampled_frames = group_frames[::step][:fn]
                        sampled_paths = group_paths[::step][:fn]
                        if len(sampled_frames) < fn:
                            sampled_frames = sampled_frames + [sampled_frames[-1]] * (fn - len(sampled_frames))
                            sampled_paths = sampled_paths + [sampled_paths[-1]] * (fn - len(sampled_paths))

                        frames_tensor = frames2tensor(
                            sampled_frames, fnum=fn, target_size=(size_t, size_t), device=self.device
                        )
                        vid_feat = self.vlm.get_vid_feat(frames_tensor)
                        vid_feat = vid_feat.detach().cpu().numpy().astype(np.float16).flatten()
                        re_feats.append(vid_feat)
                        relative_paths = [os.path.relpath(p, start=root_path).replace(os.path.sep, '/') for p in sampled_paths]
                        group_frame_paths.append(relative_paths)
                    else:
                        print(f"No keyframes in group {group_idx} for {video_name}")
            else:
                # Case 2: No txt file, group every 8 frames with specified stride
                print(f"No txt file found for {video_name}, grouping every {fn} frames with stride {stride}")
                # Use stride for overlapping groups, ensure at least fn frames remain
                frame_groups = [frames[i:i + fn] for i in range(0, max(1, len(frames) - fn + 1), stride)]
                path_groups = [images_path_list[i:i + fn] for i in range(0, max(1, len(images_path_list) - fn + 1), stride)]

                for group_idx, group_frames in enumerate(frame_groups):
                    if len(group_frames) > 0:
                        if len(group_frames) < fn:
                            group_frames = group_frames + [group_frames[-1]] * (fn - len(group_frames))
                            path_groups[group_idx] = path_groups[group_idx] + [path_groups[group_idx][-1]] * (fn - len(path_groups[group_idx]))

                        frames_tensor = frames2tensor(
                            group_frames, fnum=fn, target_size=(size_t, size_t), device=self.device
                        )
                        vid_feat = self.vlm.get_vid_feat(frames_tensor)
                        vid_feat = vid_feat.detach().cpu().numpy().astype(np.float16).flatten()
                        re_feats.append(vid_feat)
                        relative_paths = [os.path.relpath(p, start=root_path).replace(os.path.sep, '/') for p in path_groups[group_idx]]
                        group_frame_paths.append(relative_paths)
                    else:
                        print(f"No keyframes in group {group_idx} for {video_name}")
      

            # Save embeddings and frame paths
            if re_feats:
                npy_dir = os.path.join(root_path, subdir, "internvideo_npy", video_name)
                os.makedirs(npy_dir, exist_ok=True)
                npy_path = os.path.join(npy_dir, "feats.npy")
                np.save(npy_path, np.array(re_feats))
                print(f"Saved embeddings to {npy_path}")

                # Save frame paths to JSON
                json_path = os.path.join(npy_dir, "frame_paths.json")
                with open(json_path, "w") as f:
                    json.dump(group_frame_paths, f, indent=4)
                print(f"Saved frame paths to {json_path}")

    def create_bin(self, input_npy_path: str, output_bin_path: str, method="cosine", feature_shape=512):
        """Create FAISS index from .npy files."""
        if method == "L2":
            index = faiss.IndexFlatL2(feature_shape)
        elif method == "cosine":
            index = faiss.IndexFlatIP(feature_shape)
            faiss.normalize_L2 = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        else:
            raise ValueError(f"{method} not supported")

        npy_files = glob.glob(os.path.join(input_npy_path, "**/internvideo_npy/**/feats.npy"), recursive=True)
        if not npy_files:
            raise ValueError("No .npy file found, check input path!")

        id_to_path = []
        total_embeddings = 0
        for npy_file in tqdm(npy_files, desc="Building FAISS index"):
            feats = np.load(npy_file).astype(np.float32)
            num_groups, feature_dim = feats.shape
            print(f"\nProcessing {npy_file}")
            print(f"Number of groups: {num_groups}")
            print(f"Embedding dimension: {feature_dim}")
            if feature_dim != feature_shape:
                print(f"Warning: Feature dimension {feature_dim} does not match expected {feature_shape}")
                break

            if method == "cosine":
                faiss.normalize_L2(feats)
            index.add(feats)

            json_file = os.path.join(os.path.dirname(npy_file), "frame_paths.json")
            if not os.path.exists(json_file):
                print(f"Warning: {json_file} not found, skipping path mapping")
                continue
            with open(json_file, "r") as f:
                frame_paths = json.load(f)
                print(f"Frames per group (first group): {len(frame_paths[0]) if frame_paths else 0}")
            if len(frame_paths) != feats.shape[0]:
                print(f"Warning: Mismatch between embeddings ({feats.shape[0]}) and paths ({len(frame_paths)}) in {npy_file}")
                continue
            id_to_path.extend(frame_paths)
            total_embeddings += num_groups

        if total_embeddings != len(id_to_path):
            print(f"Error: FAISS index has {total_embeddings} embeddings, but id_to_path has {len(id_to_path)} groups")
            raise ValueError("Inconsistency between FAISS index and keyframe paths")
        os.makedirs(output_bin_path, exist_ok=True)
        out_path = os.path.join(output_bin_path, f"faiss_InternVideo2_{method}.bin")
        faiss.write_index(index, out_path)
        print(f"Saved FAISS index to {out_path}")

        json_path = os.path.join(output_bin_path, "keyframes_id_search_internvideo2.json")
        with open(json_path, "w") as f:
            json.dump(id_to_path, f, indent=4)
        print(f"Saved id_to_path to {json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img2npy", action='store_true', help="Convert images to .npy files")
    parser.add_argument("--npy2bin", action='store_true', help="Convert .npy files to FAISS .bin file")
    parser.add_argument('-i', required=True, type=str, help="Input keyframes dir (for img2npy) or npy dir (for npy2bin)")
    parser.add_argument('-o', required=True, type=str, help="Output bin dir (for npy2bin; ignored for img2npy)")
    parser.add_argument('--group_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--feature_shape', type=int, default=512, help="Feature dimension for FAISS index")
    parser.add_argument('--method', type=str, default="L2", choices=["L2", "cosine"], help="FAISS index method")

    args = parser.parse_args()

    if args.img2npy and not args.npy2bin:
        iv2 = InternVideo2Feats(device="cuda:1", active_model=True)
        iv2.create_npy(args.i, args.o, group_size=args.group_size, stride=args.stride)
    if args.npy2bin and not args.img2npy:
        iv2 = InternVideo2Feats(device="cuda:1", active_model=False)
        iv2.create_bin(args.i, args.o, method=args.method, feature_shape=args.feature_shape)

if __name__ == "__main__":
    main()