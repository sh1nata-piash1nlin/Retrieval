import os
import faiss
import numpy as np
import torch
import copy
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union, Dict
from googletrans import Translator
from langdetect import detect
from llava.model.builder import load_pretrained_model        # llava/model/builder.py
from llava.mm_utils import process_images, tokenizer_image_token  # llava/mm_utils.py
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN   # llava/constants.py
from llava.conversation import conv_templates                  # llava/conversation.py

class Faiss:
    def __init__(
        self,
        bin_file: str,
        model_name: str = "llava_qwen",
        pretrained: str = "zhibinlan/LLaVE-2B",
        device: str = "cuda",
        device_map: str = "auto",
        conv_template: str = "qwen_1_5",
    ):
        self.device = device
        self.conv_template = conv_template
        self.translator = Translator()
        self.tokenizer, self.model, self.image_processor, self.max_length = (
            load_pretrained_model(pretrained, None, model_name, device_map=device_map)
        )
        self.model.to(self.device).eval()

        if not os.path.isfile(bin_file):
            raise FileNotFoundError(f"FAISS index file not found: {bin_file}")
        self.index = faiss.read_index(bin_file)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            embeds = self.model.encode_multimodal_embeddings(
                tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds.cpu().numpy()

    def embed_image(self, images: List[Union[Image.Image, str]]) -> np.ndarray:
        pil_imgs = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            pil_imgs.append(img)

        img_tensors = process_images(pil_imgs, self.image_processor, self.model.config)
        img_tensors = [t.to(device=self.device, dtype=torch.float16) for t in img_tensors]
        image_sizes = [im.size for im in pil_imgs]
        conv = copy.deepcopy(conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN)
        conv.append_message(conv.roles[1], "\n")
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
                images=img_tensors,
                image_sizes=image_sizes,
            )
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds.cpu().numpy()

    def search(
        self,
        query_texts: Optional[List[str]] = None,
        query_images: Optional[List[Union[Image.Image, str]]] = None,
        top_k: int = 5,
    ) -> List[List[Dict]]:

        qvecs = []
        if query_texts:
            processed = []
            for txt in query_texts:
                try:
                    lang = detect(txt)
                except Exception:
                    lang = "en"
                if lang == "vi":
                    txt = self.translator.translate(txt, src="vi", dest="en").text
                processed.append(txt)
            qvecs.append(self.embed_text(processed))

        if query_images:
            qvecs.append(self.embed_image(query_images))

        if not qvecs:
            raise ValueError("Provide at least one of query_texts or query_images")

        Q = np.vstack(qvecs).astype(np.float32)
        distances, indices = self.index.search(Q, top_k)

        results = []
        for dists, idxs in zip(distances, indices):
            hits = []
            for idx, score in zip(idxs, dists):
                hits.append({"id": int(idx), "score": float(score)})
            results.append(hits)
        return results
