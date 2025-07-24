import torch
import torch.nn.functional as F
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from transformers import AutoTokenizer
from PIL import Image


class LLaVE2B:
    def __init__(self, model_name="zhibinlan/LLaVE-2B", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_name, 
            None,                 # no instruction template
            "llava_qwen",
            device_map="auto"     
        )
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> torch.FloatTensor:
        img_tensor = process_images([image], self.image_processor, self.model.config).to(self.device)
        vision_outputs = self.model.vision_tower(img_tensor, return_dict=True)
        img_emb = vision_outputs.image_embeds      # shape (1, D)
        return F.normalize(img_emb, p=2, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.model.config.max_seq_length
        ).to(self.device)
        text_outputs = self.model.text_encoder(
            input_ids=inputs.input_ids, 
            attention_mask=inputs.attention_mask,
            return_dict=True
        )
        txt_emb = text_outputs.last_hidden_state[:, 0, :]  
        return F.normalize(txt_emb, p=2, dim=-1)

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    # a,b: l2‑normalized
    return float((a @ b.T).item())

if __name__ == "__main__":
    model = LLaVE2B()
    img = Image.open("path/to/your/image.jpg").convert("RGB")
    caption = "A cute puppy playing in the park."
    img_emb = model.encode_image(img)    #(1, D)
    txt_emb = model.encode_text(caption) #(1, D)

    sim = cosine_similarity(img_emb, txt_emb)
    print(f"Cosine similarity between image and text: {sim:.4f}")
