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
from multi_modality.utils.utils_impl import setup_internvideo2, retrieve_text, _frame_from_video

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
class InternVideo2Model():
    def __init__(self, device: str, *args) -> None:
        self.device = device
        self.__model, self.__config = self.load_model()

    def load_model(self):
        """Load the InternVideo2 model and its configuration."""
        # Load configuration
        config_path = "/workspace/huy_aichallenge/models/InternVideo/InternVideo2/multi_modality/internvideo2_stage2_config.py"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found")
        config = Config.from_file(config_path)
        config = eval_dict_leaf(config)

        # Set model checkpoint path and device
        model_pth = "/workspace/huy_aichallenge/models/InternVideo/InternVideo2/multi_modality/weights/InternVideo2-stage2_1b-224p-f4.pt"
        if not os.path.exists(model_pth):
            raise FileNotFoundError(f"Model checkpoint {model_pth} not found")
        config['pretrained_path'] = model_pth
        config['device'] = str(self.device)

        # Load model and tokenizer
        model, tokenizer = setup_internvideo2(config)
        self.tokenizer = tokenizer  
        return model, config

    def text_encoder(self, text: str):
        text_feat = self.__model.get_txt_feat(text).cpu().detach().numpy().astype(np.float32)
        return text_feat

    def image_encoder(self, image_path: str):
        fn = self.__config.get('num_frames', 8)
        size_t = self.__config.get('size_t', 224)
        video_path = image_path
        video = cv2.VideoCapture(video_path)
        video_frames = [x for x in _frame_from_video(video)]

        frames_tensor = frames2tensor(video_frames, fnum=fn, target_size=(size_t, size_t), device=self.device)
        vid_feat = self.__model.get_vid_feat(frames_tensor)

        # vid_feat /= vid_feat.norm(dim=-1, keepdim=True)
        vid_feat = vid_feat.detach().cpu().numpy().astype(np.float16).flatten()

        return vid_feat