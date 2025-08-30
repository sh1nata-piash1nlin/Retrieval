import os
import faiss
import numpy as np
import torch
import copy
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union, Dict
from models.InternVideo.InternVideo2.multi_modality.model import InternVideo2Model, _frame_from_video, frames2tensor
from models.FDP.fdp import CustomCLIP
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from utils.query_processing import Translation
import json
import clip
from models.blip2 import BLIP2Model

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
class Faiss:
    def __init__(
        self,
        bin_files: List[str],
        dict_jsons: List[str],
        model_types: List[str],  # New parameter to specify model type per index
        device: str = "cuda:1",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.indexes = [self.load_bin_file(path) for path in bin_files]
        self.keyframes_id_searchs = [self._read_json(path, model_type) for path, model_type in zip(dict_jsons, model_types)]
        self.model_types = model_types  # Store model types for each index
        try:
            self.translate = Translation(from_lang='vi', to_lang='en', mode='googletrans')
            if not callable(self.translate):
                print(f"Translation object is not callable: {type(self.translate)}")
                self.translate = None
            else:
                print("Translation initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Translation: {e}")
            self.translate = None
        # Initialize InternVideo2Model
        self.internvideo_model = InternVideo2Model(device=self.device)
        # Initialize SigLIP model
        self.siglip_ckpt = "google/siglip2-base-patch16-512"
        self.siglip_model = AutoModel.from_pretrained(self.siglip_ckpt).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_ckpt)
        # Initialize CustomCLIP (FDP) model
        self.clip_model, _ = clip.load('RN50', device='cuda')
        self.clip_model = self.clip_model.float().eval()
        self.fdp_model = CustomCLIP('cuda', self.clip_model)
        self.fdp_model.load_state_dict(torch.load('models/FDP/RN50_reformulated.pth', map_location='cuda'), strict=False)
        self.fdp_model = self.fdp_model.eval()
        self.fdp_preprocess = self._get_preprocess(input_resolution=640)
        if "blip2" in model_types:
            self.blip2_model = BLIP2Model(device=self.device)
        else:
            self.blip2_model = None
        self.dimension = self.indexes[0].d if self.indexes else None
    def _get_preprocess(self, input_resolution=640):
        """Returns the preprocessing pipeline for CustomCLIP images."""
        return Compose([
            Resize(input_resolution, interpolation=BICUBIC),
            CenterCrop(input_resolution),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def load_bin_file(self, bin_file: str):
        """Load a FAISS index from a binary file."""
        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"FAISS index file {bin_file} not found")
        return faiss.read_index(bin_file)

    def _read_json(self, file_json: str, model_type: str):
        """Read JSON metadata file and handle different formats based on model type."""
        if not os.path.exists(file_json):
            raise FileNotFoundError(f"JSON metadata file {file_json} not found")
        with open(file_json, "r") as file:
            data = json.load(file)
        
        if model_type == "internvideo2":
            # For InternVideo2, expect nested lists where each sublist is a scene
            if isinstance(data, list) and all(isinstance(item, list) for item in data):
                return {"paths": data}  # Keep as list of lists (scenes)
            else:
                raise ValueError(f"Expected nested list for InternVideo2 JSON in {file_json}")
        else:
            # For SigLIP/FDP, expect a flat list of single frame paths
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                return {"paths": data}  # Single list of frame paths
            else:
                raise ValueError(f"Expected flat list of strings for {model_type} JSON in {file_json}")

    def embed_text(self, text: str, model_type: str = "internvideo2") -> np.ndarray:
        """Generate embedding for a single text query using specified model."""
        if model_type == "siglip2":
            inputs = self.siglip_processor(text=text, padding="max_length",return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.siglip_model.get_text_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # Normalize L2
            return feats.cpu().numpy().astype(np.float32).flatten()
        elif model_type == "fdp":
            with torch.no_grad():
                feats = self.fdp_model.embed_text(text)
                return feats.cpu().numpy().astype(np.float32).flatten()
        elif model_type == "internvideo2":
            return self.internvideo_model.text_encoder(text).astype(np.float32)
        elif model_type == "blip2":
            if self.blip2_model is None:
                raise ValueError("BLIP2Model is not initialized")
            return self.blip2_model.text_encoder(text).astype(np.float32)
        else:
            raise ValueError("model_type must be 'siglip2', 'fdp', or 'internvideo2'")

    def embed_image(self, image: Union[Image.Image, str], model_type: str = "siglip2") -> np.ndarray:
        """Generate embedding for a single image/video using specified model."""
        if model_type == "siglip2":
            if isinstance(image, str):
                image = load_image(image)
            with torch.no_grad():
                inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
                feats = self.siglip_model.get_image_features(**inputs)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
                return feats.cpu().numpy().astype(np.float32).flatten()
        elif model_type == "fdp":
            if isinstance(image, str):
                image = Image.open(image)
            with torch.no_grad():
                inputs = self.fdp_preprocess(image).unsqueeze(0).to(self.device)
                feats = self.fdp_model.embed_image(inputs)
                return feats.cpu().numpy().astype(np.float32).flatten()
        elif model_type == "internvideo2":
            if isinstance(image, Image.Image):
                img_array = np.array(image.convert("RGB"))
                video_frames = [img_array]  # Treat as single-frame video
                fn = self.internvideo_model._InternVideo2Model__config.get('num_frames', 8)
                size_t = self.internvideo_model._InternVideo2Model__config.get('size_t', 224)
                frames_tensor = frames2tensor(video_frames, fnum=fn, target_size=(size_t, size_t), device=self.internvideo_model.device)
                vid_feat = self.internvideo_model._InternVideo2Model__model.get_vid_feat(frames_tensor)
                emb = vid_feat.detach().cpu().numpy().astype(np.float32).flatten()
            elif isinstance(image, str):
                emb = self.internvideo_model.image_encoder(image)
                emb = emb.astype(np.float32)
            else:
                raise ValueError("Query image must be a PIL.Image or a file path")
            return emb
        elif model_type == "blip2":
            if self.blip2_model is None:
                raise ValueError("BLIP2Model is not initialized")
            if isinstance(image, Image.Image):
                # Save temporary image to disk since BLIP2Model.image_encoder expects a file path
                temp_path = "temp_image.jpg"
                image.save(temp_path)
                emb = self.blip2_model.image_encoder(temp_path)
                os.remove(temp_path)  # Clean up
                return emb.astype(np.float32)
            elif isinstance(image, str):
                return self.blip2_model.image_encoder(image).astype(np.float32)
            else:
                raise ValueError("Query image must be a PIL.Image or a file path")
        else:
            raise ValueError("model_type must be 'siglip2', 'fdp', or 'internvideo2'")

    def search_text(
        self,
        query_text: str,
        top_k: int = 5,
        model_type: str = "internvideo2",
    ) -> List[List[Dict]]:
        """Search for items similar to a single text query across FAISS indices."""
        if not query_text:
            raise ValueError("A text query must be provided")

        try:
            from langdetect import detect
            lang = detect(query_text)
        except Exception:
            lang = "en"
        if lang == "vi":
            query_text = self.translate(text=query_text)
        print(query_text, model_type)
        vec = self.embed_text(query_text, model_type=model_type)
        # If it's a torch tensor → convert properly
        if isinstance(vec, torch.Tensor):
            vec = vec.detach().cpu().numpy()

        # Ensure shape (1, d), not (1,1,d)
        if vec.ndim == 1:
            qvec = vec[np.newaxis, :].astype(np.float32)
        elif vec.ndim == 2:
            qvec = vec.astype(np.float32)
        else:
            raise ValueError(f"Unexpected embedding shape: {vec.shape}")

        results = []

        for idx, (index, metadata, idx_model_type) in enumerate(zip(self.indexes, self.keyframes_id_searchs, self.model_types)):
            if idx_model_type != model_type:
                continue  # Skip indices that don't match the requested model_type
            distances, indices = index.search(qvec, top_k)
            print(indices.shape)
            if model_type == "fdp":
                logit_scale = self.fdp_model.logit_scale.exp()
                distances = torch.tensor(distances, device='cuda') * logit_scale
                distances = distances.softmax(dim=-1).detach().cpu().numpy()

            hits = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(metadata["paths"]):
                    if model_type == "internvideo2":
                        # Keep the whole scene as one hit (list of frame paths)
                        hit = {
                            "id": int(idx),
                            "score": float(dist),
                            "paths": metadata["paths"][int(idx)]  # full list of frame paths
                        }
                        hits.append(hit)
                    else:
                        # Return single frame path
                        hit = {
                            "id": int(idx),
                            "score": float(dist),
                            "path": metadata["paths"][int(idx)]
                        }
                        hits.append(hit)
                else:
                    hit = {
                        "id": int(idx),
                        "score": float(dist),
                        "path": "unknown"
                    }
                    hits.append(hit)

            results.append(hits)
        
        return results

    def search_image(
        self,
        query_image: Union[Image.Image, str],
        top_k: int = 5,
        model_type: str = "siglip2",
    ) -> List[List[Dict]]:
        """Search for items similar to a single image/video query across FAISS indices."""
        if not query_image:
            raise ValueError("An image or video query must be provided")

        qvec = self.embed_image(query_image, model_type=model_type)[np.newaxis, :].astype(np.float32)
        results = []

        for idx, (index, metadata, idx_model_type) in enumerate(zip(self.indexes, self.keyframes_id_searchs, self.model_types)):
            if idx_model_type != model_type:
                continue  # Skip indices that don't match the requested model_type
            distances, indices = index.search(qvec, top_k)

            if model_type == "fdp":
                logit_scale = self.fdp_model.logit_scale.exp()
                distances = torch.tensor(distances, device=self.device) * logit_scale
                distances = distances.softmax(dim=-1).cpu().numpy()

            hits = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(metadata["paths"]):
                    if model_type == "internvideo2":
                        hit = {
                            "id": int(idx),
                            "score": float(dist),
                            "path": metadata["paths"][int(idx)]  
                        }
                        hits.append(hit)
                    else:
                        # Return single frame path
                        hit = {
                            "id": int(idx),
                            "score": float(dist),
                            "path": metadata["paths"][int(idx)]
                        }
                        hits.append(hit)
                else:
                    hit = {
                        "id": int(idx),
                        "score": float(dist),
                        "path": "unknown"
                    }
                    hits.append(hit)

            results.append(hits)
        
        return results

def main():
    # Example usage
    faiss_search = Faiss(
        bin_files=["path/to/internvideo_index.bin", "path/to/siglip_index.bin"],
        dict_jsons=["path/to/internvideo_keyframes.json", "path/to/siglip_keyframes.json"],
        model_types=["internvideo2", "siglip"],
        device="cuda"
    )
    # Text search example
    results = faiss_search.search_text("a person running", top_k=5, model_type="internvideo2")
    print("InternVideo2 text search results:", results)
    results = faiss_search.search_text("a person running", top_k=5, model_type="siglip")
    print("SigLIP text search results:", results)
    # Image search example
    results = faiss_search.search_image("path/to/query_image.jpg", top_k=5, model_type="siglip")
    print("SigLIP image search results:", results)

if __name__ == "__main__":
    main()