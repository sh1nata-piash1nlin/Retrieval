import json
import faiss
import numpy as np
import networkx as nx
from tqdm import tqdm 

def build_graph(json_file, faiss_index_file, top_k=5):
    """
    Build scene-frame graph with temporal + semantic edges
    using FAISS vectors directly.
    """
    # ---- Load mapping ----
    with open(json_file, "r") as f:
        id_to_path = json.load(f)  # list of str (frame) or list[str] (scene)

    # ---- Build base graph ----
    G = nx.Graph()
    scene_counter = 0

    for entry in tqdm(id_to_path, desc="Adding frame/scene edges"):
        if isinstance(entry, str):
            G.add_node(entry, type="frame", path=entry)
        elif isinstance(entry, list):
            scene_id = f"scene_{scene_counter}"
            scene_counter += 1
            G.add_node(scene_id, type="scene")

            for frame in entry:
                G.add_node(frame, type="frame", path=frame)
                G.add_edge(scene_id, frame, relation="contains")

            for f1, f2 in zip(entry, entry[1:]):
                G.add_edge(f1, f2, relation="temporal")

    scenes = [n for n, attr in G.nodes(data=True) if attr["type"] == "scene"]
    for s1, s2 in zip(scenes, scenes[1:]):
        G.add_edge(s1, s2, relation="temporal")

    # ---- Load FAISS index ----
    index = faiss.read_index(faiss_index_file)

    # Reconstruct all vectors
    all_features = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    # ---- Semantic edges ----
    for i, entry in enumerate(tqdm(id_to_path, desc="Adding semantic edges")):
        node_id = entry if isinstance(entry, str) else f"scene_{i}"
        if not G.has_node(node_id):
            continue

        emb = all_features[i].astype("float32").reshape(1, -1)
        D, I = index.search(emb, top_k + 1)

        for j, neighbor_idx in enumerate(I[0]):
            if neighbor_idx == i:
                continue
            neighbor_entry = id_to_path[neighbor_idx]
            neighbor_id = neighbor_entry if isinstance(neighbor_entry, str) else f"scene_{neighbor_idx}"

            if G.has_node(neighbor_id):
                G.add_edge(node_id, neighbor_id, relation="semantic", score=float(D[0][j]))

    return G

G = build_graph(
    json_file="/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/data_aichallenge2025/output_bin/keyframes_id_search_pecore_scene_frame.json",
    faiss_index_file="/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/data_aichallenge2025/output_bin/faiss_pecore_scene_frame.bin",
    top_k=5
)
nx.write_gpickle(G, "/media/hoangtv/0f9d3910-0ff9-406c-92e1-c2c8170ca6f41/data_aichallenge2025/output_bin/scene_frame_graph.gpickle")

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
