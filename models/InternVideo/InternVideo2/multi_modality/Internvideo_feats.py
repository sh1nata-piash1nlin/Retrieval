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

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# from .utils.config_impl import Config, eval_dict_leaf
# from .utils.utils_impl import setup_internvideo2
from .utils.config_impl import Config, eval_dict_leaf
from .utils.utils_impl import setup_internvideo2

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
    """Group images by segment indices."""
    images_index_array = np.array([x.split('/')[-1].replace('.jpg', '') for x in images_path_list]).astype(int)
    images_path_list = np.array(images_path_list)
    segment_images_path_list, segment_frame_list = [], []
    for segment_index in segment_index_list:
        segment_images_path = images_path_list[
            np.where((images_index_array >= segment_index[0]) & (images_index_array <= segment_index[1]))
        ]
        segment_images_path_list.append(segment_images_path)
        frames = [cv2.imread(str(img_path)) for img_path in segment_images_path]
        segment_frame_list.append(frames)
    return segment_images_path_list, segment_frame_list

class InternVideo2Feats:
    def __init__(self, device: str, active_model=True):
        self.device = device
        if active_model:
            self.config = Config.from_file("multi_modality/internvideo2_stage2_config.py")
            config = eval_dict_leaf(self.config)
            model_pth = "weights/InternVideo2-stage2_1b-224p-f4.pt"
            self.config['pretrained_path'] = model_pth
            intern_model, self.tokenizer = setup_internvideo2(config)
            self.vlm = intern_model.to(device)
            print("Img2npy mode ----- Load model successfully")
        else:
            print("Npy2bin mode")

    def create_npy(self, input_data_path: str, output_npy_dir: str, group_size=3, stride=1):
        """Create .npy files for groups of scenes."""
        fn = self.config.get('num_frames', 8)
        size_t = self.config.get('size_t', 224)

        # FIX: directly use input_data_path (no extra "images/")
        if not os.path.exists(input_data_path):
            print(f"ERROR: {input_data_path} does not exist")
            return

        video_folders = [f for f in os.listdir(input_data_path) if os.path.isdir(os.path.join(input_data_path, f))]
        for keyframe_folder in tqdm(video_folders, desc="Processing videos"):
            folder_path = os.path.join(input_data_path, keyframe_folder)
            txt_path = os.path.join(input_data_path, f"{keyframe_folder}.txt")

            if not os.path.exists(txt_path):
                print(f"Warning: {txt_path} not found, skip")
                continue

            images_path = glob.glob(os.path.join(folder_path, "*.jpg"))
            images_path_list = sorted(images_path, key=lambda x: x.split('/')[-1].replace('.jpg', ''))

            # Read scene boundaries
            segment_index_list = read_txt(txt_path)
            scene_groups = get_scene_groups(segment_index_list, group_size=group_size, stride=stride)
            # print(scene_groups)
            # Get images and frames for all scenes
            _, segment_frame_list = segment_images_by_segment_index(images_path_list, segment_index_list)

            re_feats = []
            for group_idx, scene_group in enumerate(scene_groups):
                group_frames = []
                for scene_idx in scene_group:
                    scene_frames = segment_frame_list[segment_index_list.index(scene_idx)]
                    group_frames.extend(scene_frames)

                if len(group_frames) > 0:
                    # Uniform sampling to fn frames
                    step = max(1, len(group_frames) // fn)
                    sampled_frames = group_frames[::step][:fn]

                    frames_tensor = frames2tensor(
                        sampled_frames,fnum=fn, target_size=(size_t, size_t), device=self.device
                    )
                    vid_feat = self.vlm.get_vid_feat(frames_tensor)
                    vid_feat = vid_feat.detach().cpu().numpy().astype(np.float16).flatten()
                    re_feats.append(vid_feat)
                else:
                    print(f"No keyframes in group {group_idx} for {keyframe_folder}")


            # Save embeddings
            npy_dir = os.path.join(output_npy_dir, f"{keyframe_folder}_group_{group_size}_stride_{stride}")
            os.makedirs(npy_dir, exist_ok=True)
            np.save(os.path.join(npy_dir, "feats.npy"), np.array(re_feats))
            print(f"Saved embeddings to {npy_dir}/feats.npy")

    def create_bin(self, input_npy_path: str, output_bin_path: str, method="cosine", feature_shape=512):
        """Create FAISS index from .npy files."""
        if method == "L2":
            index = faiss.IndexFlatL2(feature_shape)
        elif method == "cosine":
            index = faiss.IndexFlatIP(feature_shape)
        else:
            raise ValueError(f"{method} not supported")

        npy_files = glob.glob(os.path.join(input_npy_path, "**/feats.npy"), recursive=True)
        if not npy_files:
            raise ValueError("No .npy file found, check input path!")

        for npy_file in tqdm(npy_files, desc="Building FAISS index"):
            feats = np.load(npy_file).astype(np.float32)
            index.add(feats)

        os.makedirs(output_bin_path, exist_ok=True)
        out_path = os.path.join(output_bin_path, f"faiss_InternVideo2_{method}.bin")
        faiss.write_index(index, out_path)
        print(f"Saved FAISS index to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img2npy", action='store_true')
    parser.add_argument("--npy2bin", action='store_true')
    parser.add_argument('-i', required=True, type=str, help="Input keyframes dir (for img2npy) or npy dir (for npy2bin)")
    parser.add_argument('-o', required=True, type=str, help="Output npy or bin dir")
    parser.add_argument('--group_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()

    if args.img2npy and not args.npy2bin:
        iv2 = InternVideo2Feats(device="cuda", active_model=True)
        iv2.create_npy(args.i, args.o, group_size=args.group_size, stride=args.stride)

    if args.npy2bin and not args.img2npy:
        iv2 = InternVideo2Feats(device="cuda", active_model=False)
        iv2.create_bin(args.i, args.o)

if __name__ == "__main__":
    main()
