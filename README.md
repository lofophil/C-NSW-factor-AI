# NSW C-Factor AI — 新南威尔士州 RUSLE C 因子深度学习超分辨率

## 项目简介

利用 Sentinel-2 10m 遥感数据，通过物理约束深度学习方法，将新南威尔士州（NSW）
RUSLE 土壤侵蚀方程中的植被覆盖管理因子（C 因子）从 MODIS 500m 分辨率提升至 10m 分辨率。

**研究意义**

| 指标 | 现有产品 | 本研究目标 |
|------|---------|-----------|
| 空间分辨率 | MODIS 500m | Sentinel-2 10m |
| 时间分辨率 | 月度/年度 | 月度 |
| 覆盖区域 | NSW 全州 | NSW 全州 |
| 方法 | 经验公式 | 物理约束深度学习 |

---

## 数据来源

### 特征层（输入）
- **Sentinel-2 ARD**：通过 [DEA Sandbox](https://sandbox.dea.ga.gov.au) 获取
  - 波段：B2, B3, B4, B8, B8A, B11（10m/20m）
  - 衍生指数：NDVI, NDWI, BSI, SAVI

### 标签层（目标）
- **RUSLE C 因子栅格**：[NSW SEED 平台](https://datasets.seed.nsw.gov.au/dataset/annual-hillslope-erosion-rulse-factors)
- **月度裸土侵蚀量 GeoTIFF**：NSW SEED（2000 年至今）
- **DEA 分数植被覆盖（Fractional Cover）**：30m Landsat，1986 年至今

### 辅助层（协变量）
- **BoM AWAP 降雨**：5km，用于推算时变 R 因子
- **SRTM / Copernicus DEM**：30m，计算 LS 因子
- **SLGA 土壤格网**：90m，K 因子辅助
- **NSW 火灾范围图**：矢量，识别干扰事件

---

## 项目结构

```
nsw-cfactor-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml          # 区域范围、时间范围、超参数
├── data/
│   ├── raw/                  # 原始下载（已加入 .gitignore）
│   └── processed/            # 处理后数据（已加入 .gitignore）
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_label_preparation.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py        # 数据下载与加载工具
│   ├── indices.py           # 遥感指数计算
│   ├── preprocessing.py     # 预处理流水线
│   ├── model.py             # 模型定义
│   ├── train.py             # 训练脚本
│   └── metrics.py           # 评估指标
├── models/                   # 训练权重（用 Git LFS）
└── results/
    ├── figures/             # 可视化输出
    └── reports/             # 精度评估报告
```

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/nsw-cfactor-ai.git
cd nsw-cfactor-ai
```

### 2. 创建虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置研究区域

编辑 `configs/config.yaml`，设置 NSW 研究区范围和时间段。

### 5. 运行数据下载笔记本

```bash
jupyter lab notebooks/01_data_download.ipynb
```

---

## 研究阶段

- [x] 阶段 1：项目骨架与 GitHub 初始化
- [ ] 阶段 2：数据获取（Sentinel-2 + RUSLE 标签）
- [ ] 阶段 3：特征工程与标签对齐
- [ ] 阶段 4：模型训练（基线 RF → CNN/U-Net）
- [ ] 阶段 5：精度评估与产品输出

---

## 专利状态

本研究方法已/将向 IP Australia 提交临时专利申请。
发明名称：**用于 NSW 地表侵蚀遥感监测的物理约束深度学习方法及系统**

---

## 许可证

代码：Apache 2.0 | 数据：遵循各来源协议（DEA CC BY 4.0，NSW SEED CC BY 4.0）
