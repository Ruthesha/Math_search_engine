import sympy as sp
from functools import lru_cache
from .utils import detect_category

def canonicalize_trig(expr):
    expr = expr.rewrite(sp.sin).rewrite(sp.cos)
    expr = sp.trigsimp(sp.simplify(expr))
    expr = sp.trigsimp(expr, method="fu")
    return expr

def canonicalize_algebra(expr):
    expr = sp.expand(expr)
    expr = sp.simplify(expr)
    try:
        expr = sp.nsimplify(expr, rational=True)
    except Exception:
        pass
    return sp.simplify(expr)

def canonicalize_exp_log(expr):
    expr = expr.rewrite(sp.exp).rewrite(sp.log)
    expr = sp.simplify(expr)
    try: expr = sp.logcombine(expr, force=True)
    except: pass
    expr = sp.simplify(expr)
    return expr

def canonicalize_calculus(expr):
    try: expr = expr.doit()
    except: pass
    expr = sp.simplify(expr)
    return expr

def canonicalize(expr):
    category = detect_category(expr)

    if category == "trigonometric":
        return canonicalize_trig(expr)
    if category == "exp_log":
        return canonicalize_exp_log(expr)
    if category == "calculus":
        return canonicalize_calculus(expr)
    return canonicalize_algebra(expr)

@lru_cache(maxsize=2048)
def canonicalize_from_str(expr_str):
    env = {name: getattr(sp, name) for name in ["Integer","Rational","Symbol","Add","Mul","Pow","Abs",
                                                "sin","cos","tan","sec","csc","cot",
                                                "exp","log","sqrt","pi","E","I",
                                                "Derivative","Integral","Function"]}

    expr = eval(expr_str, env)
    return canonicalize(expr)
