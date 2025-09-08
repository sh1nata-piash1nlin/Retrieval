import os
import glob
import json
import numpy as np
from PIL import Image
import faiss
from tqdm import tqdm
import tempfile
import shutil
from transformers import AutoModelForCausalLM, AutoProcessor
from sentence_transformers import SentenceTransformer
import gc
import torch

class CaptionAndEmbeddingGenerator:
    def __init__(self, device="cuda"):
        self.device = device
        # Initialize Florence-2 model for caption generation
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            "MiaoshouAI/Florence-2-large-PromptGen-v2.0", 
            trust_remote_code=True
        ).to(device)
        self.florence_processor = AutoProcessor.from_pretrained(
            "MiaoshouAI/Florence-2-large-PromptGen-v2.0", 
            trust_remote_code=True
        )
        # Initialize Qwen3 model for embedding
        self.qwen3_model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B", 
            device=device
        )

    def generate_caption(self, image):
        """
        Generate a detailed caption for the given image using Florence-2.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Generated caption
        """
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.florence_processor(
            text=prompt, 
            images=image, 
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3
        )
        
        generated_text = self.florence_processor.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )[0]
        
        parsed_answer = self.florence_processor.post_process_generation(
            generated_text, 
            task=prompt, 
            image_size=(image.width, image.height)
        )
        
        return parsed_answer

    def generate_embedding(self, text):
        """
        Generate embedding for the given text using Qwen3.
        
        Args:
            text: str, text to encode
            
        Returns:
            np.ndarray: Normalized embedding vector
        """
        feats = self.qwen3_model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return feats[0].astype(np.float32)

    def collect_image_paths_by_folder(self, base_dir):
        """
        Collect all image paths organized by folder.
        
        Args:
            base_dir: str, path to data_aichallenge2025 directory
            
        Returns:
            dict: Mapping of (l_folder, v_folder) to list of image paths
        """
        folder_to_paths = {}
        for l_folder in os.listdir(base_dir):
            l_path = os.path.join(base_dir, l_folder, "keyframes")
            if not os.path.exists(l_path):
                continue
            for v_folder in os.listdir(l_path):
                v_path = os.path.join(l_path, v_folder)
                if not os.path.isdir(v_path):
                    continue
                image_paths = []
                for img_file in os.listdir(v_path):
                    if img_file.endswith(".jpg"):
                        img_path = os.path.join(v_path, img_file)
                        image_paths.append(img_path)
                if image_paths:
                    folder_to_paths[(l_folder, v_folder)] = sorted(image_paths)
        return folder_to_paths

    def process_keyframes(self, base_dir, output_json, index_file, batch_size=16):
        """
        Process all keyframes to generate captions, embeddings, and FAISS index.
        
        Args:
            base_dir: str, path to data_aichallenge2025 directory
            output_json: str, path to save JSON mapping
            index_file: str, path to save FAISS index
            batch_size: int, number of images to process per batch
        """
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        folder_to_paths = self.collect_image_paths_by_folder(base_dir)
        print(f"Found {len(folder_to_paths)} video folders with images.")

        global_idx = 0
        id_to_path = []
        all_features = []
        failed_images = []

        for (l_folder, v_folder), image_paths in tqdm(folder_to_paths.items(), desc="Processing folders"):
            folder_output_dir = os.path.join(base_dir, l_folder, "captions_qwen3", v_folder)
            os.makedirs(folder_output_dir, exist_ok=True)
            output_npy = os.path.join(folder_output_dir, "feats.npy")
            temp_npy = os.path.join(folder_output_dir, "feats_temp.npy")

            # Check consistency
            regenerate = False
            if os.path.exists(output_npy):
                try:
                    features = np.load(output_npy)
                    if features.shape[0] != len(image_paths):
                        print(f"Inconsistency in {v_folder}: {features.shape[0]} features vs {len(image_paths)} images. Regenerating...")
                        regenerate = True
                    del features
                    gc.collect()
                except Exception as e:
                    print(f"Error loading {output_npy}: {e}. Regenerating...")
                    regenerate = True
            else:
                print(f"No feats.npy found for {v_folder}. Generating...")
                regenerate = True

            if not regenerate:
                print(f"Skipping {v_folder}, features are consistent at {output_npy}")
                for img_path in image_paths:
                    id_to_path.append(os.path.relpath(img_path, base_dir))
                    global_idx += 1
                # Load existing features for FAISS
                try:
                    features = np.load(output_npy)
                    all_features.append(features)
                except Exception as e:
                    print(f"Error loading features for {v_folder}: {e}")
                continue

            # Process images in batches
            image_features_list = []
            valid_image_paths = []
            for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Processing images in {v_folder}"):
                batch_paths = image_paths[i:i + batch_size]
                batch_captions = []
                batch_valid_paths = []

                for img_path in batch_paths:
                    try:
                        image = Image.open(img_path)
                        caption = self.generate_caption(image)
                        batch_captions.append(caption)
                        batch_valid_paths.append(img_path)
                        image.close()
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
                        failed_images.append(img_path)
                        continue

                if batch_captions:
                    try:
                        embeddings = self.qwen3_model.encode(
                            batch_captions,
                            batch_size=batch_size,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False
                        ).astype(np.float32)
                        image_features_list.append(embeddings)
                        valid_image_paths.extend([os.path.relpath(p, base_dir) for p in batch_valid_paths])
                        global_idx += len(batch_valid_paths)
                    except Exception as e:
                        print(f"Error generating embeddings for batch in {v_folder}: {e}")
                        failed_images.extend(batch_valid_paths)

                torch.cuda.empty_cache()
                gc.collect()

            if image_features_list:
                image_features_array = np.concatenate(image_features_list, axis=0)
                np.save(temp_npy, image_features_array)
                shutil.move(temp_npy, output_npy)  # Atomic move
                print(f"Saved updated features for {v_folder} to {output_npy}")
                id_to_path.extend(valid_image_paths)
                all_features.append(image_features_array)

                del image_features_list, image_features_array
                gc.collect()

        # Save failed images log
        if failed_images:
            with open(os.path.join(base_dir, "failed_images.log"), 'w') as f:
                f.write("\n".join(failed_images))
            print(f"Logged {len(failed_images)} failed images to failed_images.log")

        # Save ID-to-path mapping atomically
        temp_json = os.path.join(os.path.dirname(output_json), "keyframes_id_search_qwen3_temp.json")
        with open(temp_json, 'w') as f:
            json.dump(id_to_path, f, indent=4)
        shutil.move(temp_json, output_json)
        print(f"Saved ID-to-path mapping to {output_json}")

        # Build FAISS index
        if all_features:
            all_features_array = np.concatenate(all_features, axis=0)
            dimension = all_features_array.shape[1]
            print(f"Feature array shape: {all_features_array.shape}")

            # Normalize features for cosine similarity
            faiss.normalize_L2(all_features_array)

            # Create FAISS index with Inner Product (IP) for cosine similarity
            index = faiss.IndexFlatIP(dimension)
            index.add(all_features_array)

            # Save FAISS index
            temp_index = os.path.join(os.path.dirname(index_file), "faiss_qwen3_L2_temp.bin")
            faiss.write_index(index, temp_index)
            shutil.move(temp_index, index_file)
            print(f"Saved FAISS index (cosine similarity) to {index_file}")

            del all_features, all_features_array, index
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print("No features were loaded for FAISS indexing.")

# Example usage
if __name__ == "__main__":
    # Define directories and files
    base_dir = "/workspace/data_aichallenge2025"
    output_json = "/workspace/data_aichallenge2025/output_bin/keyframes_id_search_caption_qwen3.json"
    index_file = "/workspace/data_aichallenge2025/output_bin/faiss_caption_qwen3_L2.bin"

    # Initialize the generator
    generator = CaptionAndEmbeddingGenerator(device="cuda")

    # Process all keyframes
    generator.process_keyframes(base_dir, output_json, index_file, batch_size=16)