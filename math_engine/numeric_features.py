import numpy as np
import sympy as sp
from .utils import detect_category, select_main_symbol

def infer_domain(expr, category):
    if category == "exp_log" and expr.has(sp.log):
        return (0.1, 5.0)
    return (-0.9, 0.9)

def numeric_profile(expr, n=64, category=None):
    if category is None:
        category = detect_category(expr)

    var = select_main_symbol(expr)
    if var is None:
        return np.full(n, float(expr.evalf()), dtype=float)

    expr_sub = expr
    dom = infer_domain(expr_sub, category)
    xs = np.linspace(dom[0], dom[1], n)

    try:
        f = sp.lambdify(var, expr_sub, "numpy")
        ys = f(xs)
    except Exception:
        ys = np.array([float(expr_sub.subs(var, x).evalf()) for x in xs])

    if np.iscomplexobj(ys):
        ys = np.real(ys)

    ys = np.where(np.isfinite(ys), ys, 0.0)
    return ys.astype(float)

def fourier_signature(values, k=None):
    values = values - np.mean(values)
    spec = np.abs(np.fft.rfft(values))
    if k: spec = spec[:k]
    norm = np.linalg.norm(spec)
    return spec / norm if norm else np.zeros_like(spec)
