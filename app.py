import random
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ----------------- Config -----------------
BASE_DIR = Path(__file__).resolve().parent
# Put your keyframes under static so Flask can serve them
# e.g. <project>/static/keyFrame/Keyframes_L01/L01_V001/000001.jpg
KEYFRAME_ROOT = BASE_DIR / 'static' / 'keyFrame'

app = Flask(__name__)

# ----------------- Helpers -----------------

# Define the base directory for keyframes
KEYFRAMES_DIRS = [
    os.path.join(app.static_folder, 'keyframe', 'Keyframes_L01'),
    os.path.join(app.static_folder, 'keyframe', 'Keyframes_L02')
]

# Fixed color palette for borders
BORDER_COLORS = [
    '#2ecc40', '#0074d9', '#ff4136', '#b10dc9', '#ff851b', '#7fdbff', '#f012be', '#01ff70', '#ffdc00', '#001f3f',
    '#39cccc', '#3d9970', '#85144b', '#aaaaaa', '#111111', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e'
]

def get_video_folders():
    video_folders = []
    for base_dir in KEYFRAMES_DIRS:
        keyframes_path = os.path.join(base_dir, 'keyframes')
        if os.path.exists(keyframes_path):
            for video_name in sorted(os.listdir(keyframes_path)):
                video_path = os.path.join(keyframes_path, video_name)
                if os.path.isdir(video_path):
                    # base_dir_name will be 'Keyframes_L01' or 'Keyframes_L02'
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
                'image_url': f'/static/keyframe/{base_name}/keyframes/{vid}/{img}',
                'border_color': border_color
            })
        if len(results) >= num_images:
            break
    random.shuffle(results) # Shuffle to mix videos
    return results[:num_images]

# ----------------- Routes -----------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/search', methods=['POST'])
def search():
    results = sample_images_from_videos()
    return render_template('_search_results.html', results=results)

@app.route('/hierarchy_search', methods=['POST'])
def hierarchy_search():
    try:
        k = int(request.form.get('k', 30))
        k1 = int(request.form.get('k1', 5))
        if k < 1: k = 30
        if k1 < 1: k1 = 5
    except Exception:
        k, k1 = 30, 5
    num_videos = max(1, k // k1)
    video_folders = get_video_folders()
    random.shuffle(video_folders)
    selected_videos = video_folders[:num_videos]
    results = []
    color_idx = 0
    for vid, path, base_name in selected_videos:
        images = [f for f in os.listdir(path) if f.lower().endswith('.jpg')]
        if not images:
            continue
        chosen = random.sample(images, min(k1, len(images)))
        color = BORDER_COLORS[color_idx % len(BORDER_COLORS)]
        color_idx += 1
        for img in chosen:
            results.append({
                'video_id': vid,
                'frame_num': os.path.splitext(img)[0],
                'image_url': f'/static/keyframe/{base_name}/keyframes/{vid}/{img}',
                'border_color': color
            })
    results = results[:k]
    results.sort(key=lambda x: x['video_id'])
    return render_template('_search_results.html', results=results)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=2714, debug=True)  