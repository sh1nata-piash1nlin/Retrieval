import os
import faiss
import numpy as np
from typing import List, Dict

class Faiss:
    def __init__(
        self,
        bin_file: str,
    ):

        if not os.path.isfile(bin_file):
            raise FileNotFoundError(f"FAISS index file not found: {bin_file}")
        self.index = faiss.read_index(bin_file)


    def search(
        self,
        query_vecs: np.ndarray,
        top_k: int = 5,
    ) -> List[List[Dict]]:

        if query_vecs is None or not isinstance(query_vecs, np.ndarray):
            raise ValueError("Provide query_vecs as a numpy array")
        Q = query_vecs.astype(np.float32)
        distances, indices = self.index.search(Q, top_k)

        results = []
        for dists, idxs in zip(distances, indices):
            hits = []
            for idx, score in zip(idxs, dists):
                hits.append({"id": int(idx), "score": float(score)})
            results.append(hits)
        return results
