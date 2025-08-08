from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment_utils import drop_outliers, trimmed_mean

_MODEL = "ProsusAI/finbert"
_tok = None
_model = None

def _load():
    global _tok, _model
    if _tok is None or _model is None:
        _tok = AutoTokenizer.from_pretrained(_MODEL)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL)
        _model.eval()

def sentiment_score(headlines: List[str]) -> float:
    """
    Simple mean aggregation of FinBERT sentiment (pos-neg) across headlines.
    Kept for backward compatibility.
    """
    if not headlines:
        return 0.0
    _load()
    with torch.no_grad():
        inputs = _tok(headlines, padding=True, truncation=True, return_tensors="pt")
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pos = probs[:, 2]
        neg = probs[:, 0]
        scores = (pos - neg).cpu().numpy()
    return float(scores.mean())

def sentiment_robust(headlines: List[str]) -> Tuple[float, List[float], List[float]]:
    """
    Robust FinBERT aggregation.
    Returns (aggregate, kept_scores, dropped_outlier_scores).
    Steps:
      1) Score each headline as (pos - neg) in [-1,1].
      2) Drop MAD-based outliers.
      3) Compute 10% trimmed mean on remaining scores.
    """
    if not headlines:
        return 0.0, [], []
    _load()
    with torch.no_grad():
        inputs = _tok(headlines, padding=True, truncation=True, return_tensors="pt")
        outputs = _model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pos = probs[:, 2]
        neg = probs[:, 0]
        scores = (pos - neg).cpu().numpy().tolist()
    kept, dropped = drop_outliers(scores, z_thresh=2.5)
    agg = trimmed_mean(kept if kept else scores, trim=0.1)
    return float(agg), kept, dropped
