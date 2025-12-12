import sympy as sp
from functools import lru_cache

def _symbolic_equiv_exact(a, b):
    try:
        if sp.simplify(a - b) == 0:
            return True
    except:
        pass
    return False

def _symbolic_equiv_relations_raw(a, b):
    info = {"exact": _symbolic_equiv_exact(a,b)}
    return info

@lru_cache(maxsize=2048)
def symbolic_relations_from_str(a_str, b_str):
    expr_a = eval(a_str, vars(sp))
    expr_b = eval(b_str, vars(sp))
    return _symbolic_equiv_relations_raw(expr_a, expr_b)

def symbolic_score_from_relations(rel):
    return 1.0 if rel.get("exact") else 0.0
