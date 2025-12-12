import sympy as sp
from collections import Counter
import numpy as np

def ast_signature(expr):
    counts = Counter()
    for node in sp.preorder_traversal(expr):
        counts[type(node).__name__] += 1
    return counts

def structural_similarity_from_sigs(sig1, sig2):
    keys = sorted(set(sig1) | set(sig2))
    v1 = np.array([sig1[k] for k in keys], float)
    v2 = np.array([sig2[k] for k in keys], float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
