# NSW C-Factor AI — Deep Learning-Based RUSLE C-Factor Downscaling for New South Wales

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

This project applies deep learning (U-Net) to downscale the RUSLE Cover and Management Factor (C-factor) from MODIS 500m resolution to Sentinel-2 10m resolution across New South Wales (NSW), Australia. The model achieves **R² = 0.82** on the test dataset.

### Research Significance

| Item | Existing Product | This Study |
|------|-----------------|------------|
| Spatial Resolution | MODIS 500m | Sentinel-2 10m |
| Temporal Resolution | Monthly | Monthly |
| Coverage | NSW State-wide | NSW State-wide |
| Method | Empirical formula | Physics-informed deep learning (U-Net) |

---

## Citation

If you use this method, code, or results in your research, please cite as follows:

```
lofophil. (2026). NSW C-Factor AI: Deep Learning-Based RUSLE C-Factor Downscaling 
for New South Wales Using Sentinel-2 10m Imagery. 
GitHub Repository. https://github.com/lofophil/C-NSW-factor-AI
```

BibTeX format:
```bibtex
@misc{lofophil2026nswcfactor,
  author    = {lofophil},
  title     = {NSW C-Factor AI: Deep Learning-Based RUSLE C-Factor Downscaling 
               for New South Wales Using Sentinel-2 10m Imagery},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/lofophil/C-NSW-factor-AI}
}
```

---

## Data Sources

### Feature Layer (Input)
- **Sentinel-2 L2A ARD**: Accessed via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC API
  - Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR)
  - Derived indices: NDVI, NDWI, BSI, SAVI
  - Resolution: 10m
  - Time: April 2024

### Label Layer (Target)
- **Monthly C-Factor Raster**: [NSW SEED Platform](https://datasets.seed.nsw.gov.au/dataset/annual-hillslope-erosion-rulse-factors)
  - Dataset: 2021–2030 Monthly Cover and Management (C-factor)
  - Original resolution: ~500m (MODIS-derived)
  - Time: April 2024 (c_202404.tif)
  - License: CC BY 4.0

---

## Complete Training Workflow

### Stage 1 — Environment Setup
- Python 3.11 (conda environment: `cfactor`)
- PyTorch 2.5.1 + CUDA 12.1 (NVIDIA RTX 3090)
- Key packages: rasterio, geopandas, stackstac, pystac-client, segmentation-models-pytorch

### Stage 2 — Data Acquisition (`notebooks/02_feature_engineering.ipynb`)
- Connected to Microsoft Planetary Computer STAC API
- Searched Sentinel-2 L2A scenes for April 2024 (cloud cover < 5%)
- Selected top 3 dates by scene count for maximum spatial coverage
- Applied maximum-NDVI monthly compositing across dates
- Computed NDVI, NDWI, BSI, SAVI indices
- Output: Feature matrix `(16811, 18790, 7)` — 7 channels at 10m resolution

### Stage 3 — Label Preparation (`notebooks/03_label_preparation.ipynb`)
- Downloaded monthly C-factor GeoTIFF (c_202404.tif) from NSW SEED
- Clipped to study area bounding box
- Reprojected from EPSG:4326 to EPSG:32755 (UTM Zone 55S)
- Bilinear resampling to match Sentinel-2 10m grid
- Output: Label array `(16811, 18790)` — C-factor values in [0, 1]

### Stage 4 — Model Training (`notebooks/07_model_training_cfactor.ipynb`)
Four models were trained and compared:

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Random Forest | 0.0084 | 0.0040 | 0.5975 |
| XGBoost | 0.0088 | 0.0040 | 0.5519 |
| 2D CNN (5-layer + residual) | 0.0104 | 0.0041 | 0.4307 |
| **U-Net (resnet34)** | **0.0058** | **0.0024** | **0.8229** |

**Training configuration (U-Net):**
- Encoder: ResNet-34
- Patch size: 256×256 pixels
- Number of patches: 20,000
- Batch size: 32
- Epochs: 100
- Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingLR
- Loss function: MSELoss + early stopping
- Training time: ~1.8 hours on RTX 3090

### Stage 5 — NSW State-wide Prediction (`notebooks/08_nsw_prediction.ipynb`)
- Divided NSW into 10 spatial tiles (approx. 350km × 350km each)
- For each tile: downloaded Sentinel-2 → extracted features → predicted C-factor → saved GeoTIFF
- Merged all tiles into final state-wide mosaic
- Output: `NSW_cfactor_2024_04_10m.tif` (28 GB, BigTIFF format)
  - Full extent: 105,703 × 120,464 pixels
  - Coverage: Eastern NSW (144°E–154°E, 37.5°S–28°S)

---

## Limitations

1. **Computational constraints**: The RTX 3090 GPU (24GB VRAM) limits the model architecture and batch size. Larger encoders (e.g., EfficientNet-B7) cause GPU memory overflow during state-wide prediction.

2. **Western NSW coverage**: Tiles covering western NSW (141°E–144°E) have very limited Sentinel-2 coverage due to the arid/semi-arid landscape with few cloud-free scenes. The final prediction currently covers **eastern NSW only**.

3. **Temporal mismatch risk**: All features and labels are from April 2024. The model may not generalise well to other months without retraining on multi-temporal data.

4. **Label resolution**: The C-factor labels are derived from MODIS 500m products. The 10m output represents a spatial downscaling based on Sentinel-2 spectral features, not independent 10m ground truth validation.

5. **Training area**: The model was trained on a pilot study area (147°E–149°E, 34°S–32.5°S) in western Sydney agricultural region. Generalisation to the full state may introduce regional biases.

---

## Repository Structure

```
C-NSW-factor-AI/
├── README.md
├── requirements.txt
├── requirements_cfactor.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 02_feature_engineering.ipynb
│   ├── 03_label_preparation.ipynb
│   ├── 06_cfactor_label.ipynb
│   ├── 07_model_training_cfactor.ipynb
│   └── 08_nsw_prediction.ipynb
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── indices.py
│   └── metrics.py
├── models/
│   └── unet_cfactor_2024_04.pth
└── results/
    └── figures/
        ├── NSW_cfactor_vs_ndvi_2024_04.png
        ├── model_comparison_cfactor_2024_04.png
        └── NSW_cfactor_2024_04_fullstate.png
```

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/lofophil/C-NSW-factor-AI.git
cd C-NSW-factor-AI
```

### 2. Create Environment
```bash
conda create -n cfactor python=3.11 -y
conda activate cfactor
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Run Notebooks in Order
```
02_feature_engineering.ipynb   → Download Sentinel-2, compute indices
03_label_preparation.ipynb     → Download and reproject C-factor label
06_cfactor_label.ipynb         → Verify label data
07_model_training_cfactor.ipynb → Train and compare 4 models
08_nsw_prediction.ipynb        → State-wide prediction and mosaic
```

---

## License

- **Code**: Apache License 2.0
- **Data**: Subject to source licenses (NSW SEED CC BY 4.0; Sentinel-2 via Planetary Computer Terms of Use)

© 2026 lofophil. Open for academic and research use. Please cite this repository if you use the code or methodology.
