import sympy as sp
import functools

def detect_category(expr):
    if expr.has(sp.Integral, sp.Derivative, sp.Sum, sp.Limit):
        return "calculus"
    if expr.has(sp.sin, sp.cos, sp.tan):
        return "trigonometric"
    if expr.has(sp.log) or expr.has(sp.exp):
        return "exp_log"
    return "algebra"

def select_main_symbol(expr):
    free = list(expr.free_symbols)
    if not free:
        return None
    for s in free:
        if s.name == "x":
            return s
    return sorted(free, key=lambda s: s.name)[0]

def clear_all_lru_caches():
    for obj in globals().values():
        if isinstance(obj, functools._lru_cache_wrapper):
            obj.cache_clear()
