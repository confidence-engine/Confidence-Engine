from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None

def _load():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def rank_relevance(headlines: List[str], topic: str, threshold: float = 0.50) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Returns (accepted, rejected) lists, each as [(headline, score), ...],
    where score is cosine similarity to the topic embedding.
    """
    _load()
    if not headlines:
        return [], []
    texts = [topic] + headlines
    embs = _model.encode(texts, normalize_embeddings=True)
    topic_vec = embs[0]
    scores = [cosine_sim(topic_vec, embs[i+1]) for i in range(len(headlines))]
    pairs = list(zip(headlines, scores))
    accepted = [(h, s) for (h, s) in pairs if s >= threshold]
    rejected = [(h, s) for (h, s) in pairs if s < threshold]
    # Sort by score desc for accepted; asc for rejected
    accepted.sort(key=lambda x: x[1], reverse=True)
    rejected.sort(key=lambda x: x[1])
    return accepted, rejected
