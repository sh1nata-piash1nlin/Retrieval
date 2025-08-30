# import faiss
# import json
# import numpy as np
# from pathlib import Path

# # ---------------- JSON merge ----------------
# # Load batch1 map
# with open("/workspace/data_aichallenge2025/output_bin/results_batch1/keyframes_id.json", "r") as f:
#     map1 = json.load(f)

# # Load old batch2 map (still points to original webp structure)
# with open("/workspace/data_aichallenge2025/output_bin/results-batch2/keyframes_id_batch2.json", "r") as f:
#     old_map2 = json.load(f)

# # Convert batch2 paths to match JPEG keyframes like map1
# DST_ROOT = Path("keyframes")  # relative to /workspace/data_aichallenge2025
# new_map2 = []
# for old_path in old_map2:
#     parts = Path(old_path).parts[-3:]  # ("Kxx","Vxx","000016.webp")
#     Kxx, Vxx, fname = parts
#     new_path = Path(f"keyframes_Videos_{Kxx}") / "keyframes" / f"{Kxx}_{Vxx}" / Path(fname).with_suffix(".jpg")
#     new_map2.append(str(new_path))

# map2 = new_map2

# # Merge JSON
# merged_map = map1 + map2
# json_out_path = "/workspace/data_aichallenge2025/output_bin/keyframes_id_search_blip2.json"
# with open(json_out_path, "w") as f:
#     json.dump(merged_map, f, indent=2)

# # print(f"✅ Merged JSON saved with {len(merged_map)} entries at {json_out_path}")

# # # ---------------- FAISS merge ----------------
# # index1_path = "/workspace/data_aichallenge2025/output_bin/results_batch1/image_index_l2.bin"
# # index2_path = "/workspace/data_aichallenge2025/output_bin/results-batch2/image_index_l2_batch2.bin"
# # faiss_out_path = "/workspace/data_aichallenge2025/output_bin/faiss_blip2_l2.bin"

# # # Load indexes
# # index1 = faiss.read_index(index1_path)
# # index2 = faiss.read_index(index2_path)

# # # Reconstruct all vectors from index2
# # xb = index2.reconstruct_n(0, index2.ntotal)

# # # Generate shifted IDs for batch2 vectors
# # start_id = index1.ntotal
# # ids = np.arange(start_id, start_id + index2.ntotal)

# # # Add batch2 vectors with explicit IDs (IndexIDMap)
# # index1.add_with_ids(xb, ids)

# # # Save merged FAISS index
# # faiss.write_index(index1, faiss_out_path)

# # print(f"✅ Merged FAISS index saved with {index1.ntotal} vectors at {faiss_out_path}")

# # # ---------------- Sanity check ----------------
# # assert len(merged_map) == index1.ntotal, "❌ JSON paths and FAISS vectors count mismatch!"
# # print("✅ Sanity check passed: JSON and FAISS index are aligned.")

import json
import os

# Define file paths relative to huy_aichallenge directory
file1 = "../data_aichallenge2025/output_bin/metadata.json"
file2 = "../data_aichallenge2025/output_bin/metadata_batch2.json"  # Corrected potential typo in file name

# Check if files exist
if not os.path.exists(file1):
    print(f"Error: File not found: {file1}")
    exit(1)
if not os.path.exists(file2):
    print(f"Error: File not found: {file2}")
    exit(1)

# Load JSON files
with open(file1, "r") as f:
    map1 = json.load(f)
with open(file2, "r") as f:
    map2 = json.load(f)

# Merge JSON
merged_map = map1 + map2
json_out_path = "../data_aichallenge2025/output_bin/keyframes_id_search_asr.json"
with open(json_out_path, "w") as f:
    json.dump(merged_map, f, indent=2)
print(f"Merged JSON saved to {json_out_path}")