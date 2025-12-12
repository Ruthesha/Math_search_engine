import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    rationalize
)

def _normalize_euler_power(expr_str: str) -> str:
    def repl(m): return f"exp({m.group(1)})"
    return re.sub(r"e\^(\([^()]*\)|[+\-]?[a-zA-Z0-9_]+)", repl, expr_str)

def _fix_imaginary_denominator(expr_str: str) -> str:
    return re.sub(r"/\s*(\d+i)(?!\w)", lambda m: f"/({m.group(1)})", expr_str)

def _convert_abs_bars(expr_str: str) -> str:
    prev = None
    pattern = r"\|([^|]+)\|"
    while prev != expr_str:
        prev = expr_str
        expr_str = re.sub(pattern, r"Abs(\1)", expr_str)
    return expr_str

def _fix_trig_no_parentheses(expr_str: str) -> str:
    return re.sub(r"\b(sin|cos|tan|sec|csc|cot)([A-Za-z])\b", r"\1(\2)", expr_str)

def _fix_trig_power(expr_str: str) -> str:
    return re.sub(r"\b(sin|cos|tan|sec|csc|cot)\^(\d+)\s*([A-Za-z])\b",
                  r"\1(\3)**\2", expr_str)

def parse_and_normalize(expr_str: str) -> sp.Expr:
    expr_str = _fix_trig_no_parentheses(expr_str)
    expr_str = _fix_trig_power(expr_str)
    expr_str = expr_str.replace(" ", "")
    expr_str = _fix_imaginary_denominator(expr_str)
    expr_str = _convert_abs_bars(expr_str)
    expr_str = _normalize_euler_power(expr_str)

    parser_locals = {
        "pi": sp.pi, "oo": sp.oo, "Sum": sp.Sum, "Limit": sp.Limit,
        "gamma": sp.gamma, "cbrt": sp.cbrt, "sin": sp.sin, "cos": sp.cos,
        "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt
    }

    transformations = (
        standard_transformations +
        (implicit_multiplication_application, convert_xor, rationalize)
    )

    try:
        expr = parse_expr(expr_str, local_dict=parser_locals,
                          transformations=transformations)
    except Exception:
        expr = sp.sympify(expr_str)

    def is_cube_root(node):
        return (isinstance(node, sp.Pow) and
                node.exp.is_Rational and node.exp.p == 1 and node.exp.q == 3)

    expr = expr.replace(is_cube_root,
                        lambda x: sp.real_root(x.args[0], 3))

    subs = {}
    for s in expr.free_symbols:
        if s.name == "i": subs[s] = sp.I
        elif s.name == "e": subs[s] = sp.E
        else: subs[s] = sp.Symbol(s.name, real=True)

    return expr.subs(subs)
