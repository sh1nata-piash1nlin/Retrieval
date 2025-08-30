import torch
import clip
import faiss
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_preprocess(input_resolution=640):
    return Compose([
        Resize(input_resolution, interpolation=BICUBIC),
        CenterCrop(input_resolution),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CLIP_model, _ = clip.load('RN50', device=device)
CLIP_model = CLIP_model.float().eval()
model = CustomCLIP(device, CLIP_model)
model.load_state_dict(torch.load('./save/fdp-8th.pt')['model_state_dict'])
model.eval()

# Preprocess and embed images
preprocess = get_preprocess()
image_paths = ['./datasets/IIIT_STR_V1.0/imgDatabase_pad_square/image1.jpg',
               './datasets/IIIT_STR_V1.0/imgDatabase_pad_square/image2.jpg']
images = torch.cat([preprocess(Image.open(img)).unsqueeze(0) for img in image_paths]).to(device)
image_features = model.embed_image(images)

# Embed text queries
queries = ['a dog in a park', 'a cat on a mat']
text_features = model.embed_text(queries)

# FAISS search with logit_scale.exp()
dim = image_features.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(image_features.cpu().numpy().astype(np.float32))
distances, indices = index.search(text_features.cpu().numpy().astype(np.float32), k=2)

# Apply logit_scale.exp() and softmax
logit_scale = model.logit_scale.exp()
scaled_scores = torch.tensor(distances, device=device) * logit_scale
scaled_scores = scaled_scores.softmax(dim=-1)

# Print results
for i, query in enumerate(queries):
    print(f"\nQuery: {query}")
    print(f"Top 2 images (indices): {indices[i]}")
    print(f"Scaled scores: {scaled_scores[i].cpu().numpy()}")