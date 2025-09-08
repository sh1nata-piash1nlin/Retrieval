import json
import os
import faiss

# Input files
file1 = "../data_aichallenge2025/output_bin/metadata_qwen_batch1.json"
file2 = "../data_aichallenge2025/output_bin/metadata_qwen_batch2.json"
bin1 = "../data_aichallenge2025/output_bin/asr_qwenembedding_batch1.bin"
bin2 = "../data_aichallenge2025/output_bin/asr_qwenembedding_batch2.bin"

# Output
json_out_path = "../data_aichallenge2025/output_bin/keyframes_id_search_qwen3.json"
bin_out_path = "../data_aichallenge2025/output_bin/faiss_qwen3_cosine.bin"

# Check files exist
for f in [file1, file2, bin1, bin2]:
    if not os.path.exists(f):
        raise FileNotFoundError(f"File not found: {f}")

# Load metadata
with open(file1, "r", encoding="utf-8") as f:
    map1 = json.load(f)
with open(file2, "r", encoding="utf-8") as f:
    map2 = json.load(f)

merged_map = map1 + map2

# ---- Convert to list of frame path lists ----
scene_groups = []
for seg in merged_map:
    video_id = seg["video"].replace(".mp4", "")
    start, end = seg["start_frame"], seg["end_frame"]

    # Build frame paths (step = 1 frame, bạn có thể thay đổi step nếu muốn thưa hơn)
    frame_paths = [
        f"keyframes_Videos_{video_id[:3]}/keyframes/{video_id}/{frame:06d}.jpg"
        for frame in range(start, end + 1)
    ]

    if frame_paths:
        scene_groups.append(frame_paths)

# Save new JSON
with open(json_out_path, "w", encoding="utf-8") as f:
    json.dump(scene_groups, f, indent=2, ensure_ascii=False)
print(f"Merged+converted JSON saved to {json_out_path} ({len(scene_groups)} groups)")

# ---- Merge FAISS indices ----
index1 = faiss.read_index(bin1)
index2 = faiss.read_index(bin2)

if index1.d != index2.d:
    raise ValueError(f"Dimension mismatch: {index1.d} vs {index2.d}")

xb = index2.reconstruct_n(0, index2.ntotal)
index1.add(xb)

faiss.write_index(index1, bin_out_path)
print(f"Merged FAISS index saved to {bin_out_path}, total={index1.ntotal}")
