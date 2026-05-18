"""
metrics.py — 模型评估指标
包含 C 因子预测的标准评估指标和空间精度评估工具
"""

import numpy as np
from typing import Dict


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差（Root Mean Squared Error）"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均绝对误差（Mean Absolute Error）"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 R²"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum((y_t - y_t.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """系统偏差（正值=过高估计，负值=低估）"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(y_pred[mask] - y_true[mask]))


def evaluate_all(y_true: np.ndarray,
                 y_pred: np.ndarray,
                 label: str = "模型") -> Dict[str, float]:
    """
    一次性计算所有评估指标

    参数
    ----
    y_true : 真实 C 因子值（来自 RUSLE 标签）
    y_pred : 模型预测 C 因子值
    label  : 模型名称（用于打印）

    返回
    ----
    包含所有指标的字典
    """
    results = {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
    }

    print(f"\n{'='*40}")
    print(f"  评估结果 — {label}")
    print(f"{'='*40}")
    for k, v in results.items():
        print(f"  {k:<8}: {v:+.4f}")
    print(f"{'='*40}")

    return results


def improvement_over_baseline(baseline_metrics: Dict[str, float],
                               model_metrics: Dict[str, float]) -> None:
    """打印相对于基准（MODIS 500m）的改善幅度"""
    print("\n  相对于 MODIS 500m 基准的改善:")
    for metric in ["RMSE", "MAE"]:
        if metric in baseline_metrics and metric in model_metrics:
            delta = baseline_metrics[metric] - model_metrics[metric]
            pct = delta / baseline_metrics[metric] * 100
            sign = "↓" if delta > 0 else "↑"
            print(f"  {metric}: {sign} {abs(pct):.1f}%")
    if "R2" in baseline_metrics and "R2" in model_metrics:
        delta = model_metrics["R2"] - baseline_metrics["R2"]
        print(f"  R²:   +{delta:.4f}")
