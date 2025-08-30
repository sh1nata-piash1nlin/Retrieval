import numpy as np
import torch
from PIL import Image
from pathlib import Path
import sys
from PIL import Image

from lavis.models import load_model_and_preprocess

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[4]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
    

class BLIP2Model():
    def __init__(self, device: str, *args):
        self.device = device
        self.__model, self.__vis_processors, self.__text_processors = self.load_model()
        
    def load_model(self):
        model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", 
                                                                           device= self.device, is_eval=True)
        return model, vis_processors, text_processors
    
    def text_encoder(self, text: str):
        text_processed = self.__text_processors["eval"](text)
        text_feat = self.__model.extract_features({"text_input": text_processed}, mode="text").text_embeds[0,0,:] 
        text_feat = text_feat.detach().cpu().detach().numpy().astype(np.float32).reshape(1,-1)
        return text_feat
    
    def image_encoder(self, image_path: str) :
        image_processed = self.__vis_processors["eval"](Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        image_feat = self.__model.extract_features({"image": image_processed }, mode="image").image_embeds[0,0,:]
        image_feat = image_feat.detach().cpu().numpy().astype(np.float16).flatten()
        return image_feat
    
    
if __name__ == "__main__":
    txt = "A black car was crushed by a broken tree branch."
    img_path = "/home/toonies/Learn/Text_Video_Retrieval/data/images/Keyframes_L01/L01_V001/001287.jpg"
    blip2_model = BLIP2Model("cuda")
    img_feat = blip2_model.image_encoder(img_path)
    print("feature of imgage shape", img_feat.shape)
    txt_feat = blip2_model.text_encoder(txt)
    print("feature of text shape", txt_feat.shape) # (768,)

    a = img_feat
    b = txt_feat
    sim = np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))
    print("Cosine similar", sim)