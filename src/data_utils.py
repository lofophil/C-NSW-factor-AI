"""
data_utils.py — 数据加载与工具函数
处理 Sentinel-2 ARD、RUSLE 标签和辅助数据的加载、验证和保存
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import yaml


# ─────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """加载项目配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# 路径管理
# ─────────────────────────────────────────────

def get_project_root() -> Path:
    """获取项目根目录（包含 configs/ 的目录）"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "configs").exists():
            return parent
    return current.parent.parent


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，不存在则创建"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────────────────────────────
# 数组工具
# ─────────────────────────────────────────────

def normalize_band(arr: np.ndarray,
                   percentile_low: float = 2.0,
                   percentile_high: float = 98.0) -> np.ndarray:
    """
    百分位数拉伸归一化（去除异常值影响）
    将遥感波段归一化到 [0, 1]
    """
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    p_low = np.percentile(valid, percentile_low)
    p_high = np.percentile(valid, percentile_high)
    if p_high == p_low:
        return np.zeros_like(arr)
    normalized = (arr - p_low) / (p_high - p_low)
    return np.clip(normalized, 0.0, 1.0)


def mask_nodata(arr: np.ndarray,
                nodata_value: float = -9999.0,
                fill_value: float = np.nan) -> np.ndarray:
    """将无效像素替换为 NaN"""
    result = arr.astype(float).copy()
    result[arr == nodata_value] = fill_value
    return result


def check_alignment(arr1: np.ndarray, arr2: np.ndarray,
                    name1: str = "特征", name2: str = "标签") -> bool:
    """检查两个数组的空间尺寸是否一致"""
    if arr1.shape[-2:] != arr2.shape[-2:]:
        print(f"[警告] {name1} 形状 {arr1.shape} 与 {name2} 形状 {arr2.shape} 不匹配")
        return False
    return True


# ─────────────────────────────────────────────
# NSW SEED 月度侵蚀数据 URL 生成
# ─────────────────────────────────────────────

def build_seed_monthly_url(year: int, month: int) -> str:
    """
    构造 NSW SEED 月度裸土侵蚀 GeoTIFF 的下载路径规律
    实际 resource ID 需从 SEED API 查询，此处为示例格式
    """
    # 注意：实际 URL 中的 resource_id 需从 SEED 平台 API 获取
    # 参考基础格式：ero_YYYYMM.tif
    filename = f"ero_{year}{month:02d}b.tif"
    return filename


def list_available_months(start_year: int = 2019,
                           end_year: int = 2023) -> list:
    """生成年月组合列表，用于批量下载"""
    months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            months.append((year, month))
    return months


# ─────────────────────────────────────────────
# 数据摘要
# ─────────────────────────────────────────────

def print_array_info(arr: np.ndarray, name: str = "数组") -> None:
    """打印数组基本统计信息，用于数据验证"""
    valid = arr[np.isfinite(arr)]
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  形状:    {arr.shape}")
    print(f"  数据类型: {arr.dtype}")
    if len(valid) > 0:
        print(f"  最小值:  {valid.min():.4f}")
        print(f"  最大值:  {valid.max():.4f}")
        print(f"  均值:    {valid.mean():.4f}")
        print(f"  标准差:  {valid.std():.4f}")
        nan_pct = (1 - len(valid)/arr.size) * 100
        print(f"  NaN 比例: {nan_pct:.1f}%")
    else:
        print("  全部为 NaN 或无效值！")
