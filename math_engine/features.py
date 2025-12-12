import numpy as np
from .numeric_features import numeric_profile, fourier_signature
from .utils import detect_category, select_main_symbol

def build_feature(expr, n_samples=64):
    category = detect_category(expr)
    f_vals = numeric_profile(expr, n_samples, category)

    try:
        var = select_main_symbol(expr)
        if var:
            d_expr = expr.diff(var)
            df_vals = numeric_profile(d_expr, n_samples, category)
        else:
            df_vals = np.zeros_like(f_vals)
    except:
        df_vals = np.zeros_like(f_vals)

    fft_vals = fourier_signature(f_vals, k=n_samples//2)
    return np.concatenate([f_vals, df_vals, fft_vals])
