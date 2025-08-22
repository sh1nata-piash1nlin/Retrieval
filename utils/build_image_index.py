import os
import json
import faiss
import numpy as np
import torch
import gc
from PIL import Image
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy

# Set PyTorch memory allocation settings
torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of available GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

class ImageIndexBuilder:
    def __init__(
        self,
        model_name: str = "llava_qwen",
        pretrained: str = "zhibinlan/LLaVE-2B",
        device: str = "cuda",
        device_map: str = "auto",
        conv_template: str = "qwen_1_5",
    ):
        self.device = device
        self.conv_template = conv_template
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(pretrained, None, model_name, device_map=device_map)
        )
        self.model.to(self.device).eval()

    def embed_image(self, image_path: str) -> np.ndarray:
        """Embed a single image using the LLaVA model."""
        try:
            img = Image.open(image_path).convert("RGB")
            img_tensor = process_images([img], self.image_processor, self.model.config)
            img_tensor = [t.to(device=self.device, dtype=torch.float16) for t in img_tensor]
            image_size = [img.size]

            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
            conv.append_message(conv.roles[1], "\\n")
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(self.device)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

            with torch.no_grad():
                embeds = self.model.encode_multimodal_embeddings(
                    input_ids,
                    attention_mask=attention_mask,
                    images=img_tensor,
                    image_sizes=image_size,
                )
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                result = embeds.cpu().numpy()

            # Clear GPU memory
            del img_tensor, input_ids, attention_mask, embeds
            torch.cuda.empty_cache()
            gc.collect()

            return result
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def build_index(self, image_folder: str, output_bin: str, output_json: str, batch_size: int = 4):
        """Build FAISS index from images and save mapping to JSON."""
        # Get all image paths
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(Path(image_folder).glob(ext))
        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No images found in {image_folder}")

        # Process first image to get embedding dimension
        print("Getting embedding dimension...")
        sample_embedding = None
        for img_path in image_paths:
            sample_embedding = self.embed_image(str(img_path))
            if sample_embedding is not None:
                break
        
        if sample_embedding is None:
            raise ValueError("Could not process any sample image")

        dimension = sample_embedding.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Build id to path mapping
        id_to_path = {}
        current_id = 0

        # Process images in batches
        print(f"Processing {len(image_paths)} images in batches of {batch_size}...")
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_embeddings = []
            
            for img_path in batch_paths:
                embedding = self.embed_image(str(img_path))
                if embedding is not None:
                    batch_embeddings.append(embedding)
                    id_to_path[current_id] = str(img_path)
                    current_id += 1

            if batch_embeddings:
                # Add batch to index
                batch_embeddings = np.vstack(batch_embeddings)
                index.add(batch_embeddings)
                
                # Clear memory after each batch
                del batch_embeddings
                torch.cuda.empty_cache()
                gc.collect()

        # Save FAISS index
        print(f"Saving index to {output_bin}")
        faiss.write_index(index, output_bin)

        # Save mapping to JSON
        print(f"Saving mapping to {output_json}")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(id_to_path, f, ensure_ascii=False, indent=2)

        print("Done!")
        print(f"Total images indexed: {len(id_to_path)}")


if __name__ == "__main__":
    # Initialize the builder
    builder = ImageIndexBuilder()

    # Define paths
    image_folder = "static/keyframe/Keyframes_L01"
    output_bin = "static/index/keyframe_index.bin"
    output_json = "static/index/keyframe_mapping.json"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)

    # Build the index
    builder.build_index(image_folder, output_bin, output_json)
