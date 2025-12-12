import numpy as np
from dataclasses import dataclass
from collections import Counter

from .normalize import parse_and_normalize
from .canonicalize import canonicalize_from_str
from .features import build_feature
from .structural_features import ast_signature
from .symbolic_relations import symbolic_relations_from_str, symbolic_score_from_relations
from .utils import detect_category

@dataclass
class Entry:
    original: str
    canonical_str: str
    category: str
    feature: np.ndarray
    ast_sig: Counter

class EquationLibrary:
    def __init__(self, equations, n_samples=64):
        self.entries = []
        self.n_samples = n_samples

        for eq in equations:
            parsed = parse_and_normalize(eq)
            canonical = canonicalize_from_str(repr(parsed))
            canonical_str = repr(canonical)

            category = detect_category(canonical)
            feature = build_feature(canonical, n_samples)
            ast_sig = ast_signature(canonical)

            self.entries.append(
                Entry(eq, canonical_str, category, feature, ast_sig)
            )

        self.matrix = np.vstack([e.feature for e in self.entries])

    def query(self, expr, k=5):
        parsed = parse_and_normalize(expr)
        canonical = canonicalize_from_str(repr(parsed))
        canon_str = repr(canonical)

        q_feat = build_feature(canonical, self.n_samples)

        results = []
        for entry in self.entries:
            dist = np.linalg.norm(entry.feature - q_feat)

            rel = symbolic_relations_from_str(canon_str, entry.canonical_str)
            sym_score = symbolic_score_from_relations(rel)

            results.append({
                "equation": entry.original,
                "canonical_entry": entry.canonical_str,
                "symbolic": rel,
                "score": sym_score,
                "distance": dist
            })

        results.sort(key=lambda x: (-x["score"], x["distance"]))
        return results[:k]
