import random
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json
from utils.faiss import Faiss
from models.SigLip2 import embed_siglip, processor, model, embed_siglip_text
from utils.query_processing import Translation
import numpy as np
from transformers import AutoProcessor
import torch
from langdetect import detect

BASE_DIR = Path(__file__).resolve().parent
KEYFRAME_ROOT = BASE_DIR / 'static' / 'keyFrame'
app = Flask(__name__)


def get_keyframe_video_dirs():
    keyframe_root = os.path.join(app.static_folder, 'keyframe')
    if not os.path.exists(keyframe_root):
        return []
    return [os.path.join(keyframe_root, d) for d in os.listdir(keyframe_root)
            if os.path.isdir(os.path.join(keyframe_root, d)) and d.startswith('keyframes_Videos_')]

BORDER_COLORS = [
    '#2ecc40', '#0074d9', '#ff4136', '#b10dc9', '#ff851b', '#7fdbff', '#f012be', '#01ff70', '#ffdc00', '#001f3f',
    '#39cccc', '#3d9970', '#85144b', '#aaaaaa', '#111111', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e'
]

def get_video_folders():
    video_folders = []
    for base_dir in get_keyframe_video_dirs():
        # Each base_dir is like /static/keyframe/keyframes_Videos_L21
        for video_name in sorted(os.listdir(base_dir)):
            video_path = os.path.join(base_dir, video_name)
            if os.path.isdir(video_path):
                base_dir_name = os.path.basename(base_dir)
                video_folders.append((video_name, video_path, base_dir_name))
    return video_folders

def sample_images_from_videos(num_images=30, images_per_video=(3,4)):
    video_folders = get_video_folders()
    random.shuffle(video_folders)
    results = []
    border_color = '#aaaaaa'  # Grey color for all borders
    for vid, path, base_name in video_folders:
        images = [f for f in os.listdir(path) if f.lower().endswith('.jpg')]
        if not images:
            continue
        k = random.randint(*images_per_video)
        chosen = random.sample(images, min(k, len(images)))
        for img in chosen:
            results.append({
                'video_id': vid,
                'frame_num': os.path.splitext(img)[0],
                'image_url': f'/static/keyframe/{base_name}/{vid}/{img}',
                'border_color': border_color
            })
        if len(results) >= num_images:
            break
    random.shuffle(results) # Shuffle to mix videos
    return results[:num_images]


@app.route('/')
def home():
    return render_template('home.html')

SIGLIP_FAISS_BIN = str(BASE_DIR / 'faiss_siglip_L2.bin')
SIGLIP_JSON = str(BASE_DIR / 'keyframes_id_search_siglip2.json')
siglip_index = None
siglip_id_to_path = None
# try:
#     siglip_index = Faiss(SIGLIP_FAISS_BIN)
#     with open(SIGLIP_JSON, 'r') as f:
#         siglip_id_to_path = json.load(f)
#     translation = Translation(from_lang='vi', to_lang='en', mode='googletrans')
# except Exception as e:
#     print(f"SigLIP2 index or mapping not loaded: {e}")
try:
    siglip_index = Faiss(SIGLIP_FAISS_BIN)
    with open(SIGLIP_JSON, 'r') as f:
        siglip_id_to_path = json.load(f)
    index_count   = siglip_index.index.ntotal   #sanity check
    mapping_count = len(siglip_id_to_path)
    if index_count != mapping_count:
        raise RuntimeError(
            f"SigLIP index/mapping size mismatch: index={index_count}, paths={mapping_count}. "
            "Rebuild the FAISS bin and JSON together."
        )
    translation = Translation(from_lang='vi', to_lang='en', mode='googletrans')

except Exception as e:
    print(f"SigLIP2 index or mapping not loaded: {e}")
    siglip_index = None
    siglip_id_to_path = None

@app.route('/siglip2_search', methods=['POST'])
def siglip2_search():
    query = request.form.get('query')
    top_k = int(request.form.get('top_k', 30))
    if not query or not siglip_index or not siglip_id_to_path:
        return jsonify({'error': 'Missing query or index'}), 400
    try:
        lang = Translation().__call__(query) if detect(query) == 'vi' else query
    except Exception:
        lang = query
    inputs = processor(text=lang, return_tensors="pt").to(model.device)
    with torch.no_grad():
        query_vec = embed_siglip_text(lang)
    results = siglip_index.search(query_vecs=query_vec, top_k=top_k)[0]
    images = []
    for hit in results:
        idx = hit['id']
        score = hit['score']
        img_path = siglip_id_to_path[idx]
        parts = img_path.split('/')
        video_id = parts[-2] if len(parts) >= 2 else ''
        frame_num = parts[-1].split('.')[0] if len(parts) >= 1 else ''
        images.append({
            'image_url': f'/static/keyframe/{img_path}',
            'score': score,
            'video_id': video_id,
            'frame_num': frame_num,
            'border_color': '#ffa500'
        })
    return render_template('_search_results.html', results=images)

@app.route('/search', methods=['POST'])
def search():
    results = sample_images_from_videos()
    return render_template('_search_results.html', results=results)



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=2714, debug=True)  