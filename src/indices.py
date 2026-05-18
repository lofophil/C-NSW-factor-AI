"""
indices.py — 遥感指数计算
从 Sentinel-2 波段计算 NDVI、NDWI、BSI、SAVI 等与 C 因子相关的指数
"""

import numpy as np


def ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    归一化差值植被指数
    NDVI = (NIR - Red) / (NIR + Red)
    范围: [-1, 1]，值越高植被越茂密，C 因子越低
    """
    return (nir - red) / (nir + red + eps)


def ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    归一化差值水体/湿度指数
    NDWI = (Green - NIR) / (Green + NIR)
    正值表示水体或高土壤含水量
    """
    return (green - nir) / (green + nir + eps)


def bsi(blue: np.ndarray, red: np.ndarray,
        nir: np.ndarray, swir: np.ndarray,
        eps: float = 1e-8) -> np.ndarray:
    """
    裸土指数
    BSI = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
    值越高裸土比例越大，C 因子越高（抗侵蚀能力越弱）
    """
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue)
    return num / (den + eps)


def savi(red: np.ndarray, nir: np.ndarray,
         L: float = 0.5, eps: float = 1e-8) -> np.ndarray:
    """
    土壤调整植被指数（适合 NSW 稀疏植被区）
    SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
    L=0.5 为默认值，稀疏植被区可调至 0.25
    """
    return (nir - red) / (nir + red + L + eps) * (1 + L)


def calculate_all_indices(bands: dict) -> dict:
    """
    批量计算所有指数

    参数
    ----
    bands : dict，键为波段名，值为 np.ndarray
        必须包含: blue, green, red, nir, swir

    返回
    ----
    dict，包含所有计算的指数
    """
    required = {"blue", "green", "red", "nir", "swir"}
    missing = required - set(bands.keys())
    if missing:
        raise ValueError(f"缺少波段: {missing}")

    indices = {
        "NDVI": ndvi(bands["red"], bands["nir"]),
        "NDWI": ndwi(bands["green"], bands["nir"]),
        "BSI":  bsi(bands["blue"], bands["red"], bands["nir"], bands["swir"]),
        "SAVI": savi(bands["red"], bands["nir"]),
    }
    return indices


def ndvi_to_c_factor_empirical(ndvi_arr: np.ndarray) -> np.ndarray:
    """
    基于 NDVI 的 C 因子经验公式（用于基线对比）
    C = exp(-α * NDVI)，α ≈ 2.0（文献常用值）
    参考: Van der Knijff et al. (1999)

    注意：这是经验公式，本项目目标是用 AI 替代此方法
    """
    alpha = 2.0
    c = np.exp(-alpha * np.clip(ndvi_arr, 0, 1))
    return np.clip(c, 0.0, 1.0)
