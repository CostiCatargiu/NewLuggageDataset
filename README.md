<!-- =======================================================
🚨 Real-Time Abandoned Luggage Detection (YOLOv12 + Spatio-Temporal)
README.md — drop this file into your repo root.
Tip: replace placeholders like <YOUR_LINK> / <PATHS> / <PAPER>.
======================================================= -->

<div align="center">

# 🚨 Real-Time Abandoned Luggage Detection  
## Enhanced YOLOv12 + Tracking-by-Detection + Spatio-Temporal Reasoning

<!-- Badges -->
<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" />
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLOv12-111827?logo=github&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

<!-- Short description (≈350 chars) -->
<p>
Real-time abandoned luggage detection framework using enhanced YOLOv12 models and spatio-temporal reasoning. A custom-trained YOLOv12m improves small-object detection, while YOLOv12x ensures robust person tracking. Abandonment is determined through interpretable distance and time constraints for reliable public safety surveillance.
</p>

</div>

---
<img width="924" height="842" alt="image" src="https://github.com/user-attachments/assets/6b436ba9-0d92-4bc3-b14d-9e35fa155664" />

## 🎯 What this project does

Abandoned luggage detection in real surveillance video is hard because luggage items are:
- 🧳 **Small** (low pixel footprint)
- 🕶️ **Visually diverse** (bags/backpacks/trolleys)
- 👥 **Often occluded** in crowded scenes  
…and because “abandoned” requires **time + distance reasoning**, not just detection.

This repo provides a **complete pipeline**:
- **Luggage detection** → custom-trained **YOLOv12m**
- **Person detection** → **YOLOv12x** (high recall in crowds)
- **Association over time** → tracking-by-detection
- **Abandonment decision** → interpretable **radius + duration** rules

---
## 🧠 Pipeline overview
<img width="761" height="435" alt="image" src="https://github.com/user-attachments/assets/beec2c70-dd65-403a-90a1-1b331a8028ab" />

## ✨ Key contributions

- ✅ **Large-scale public dataset**: 29,053 images / 130,475 instances (bag, backpack, trolley)  
- ✅ **Small-object–aware loss (training-only change)** — YOLOv12m architecture unchanged  
- ✅ **Robust pairing** of person ↔ luggage via tracking-by-detection  
- ✅ **Interpretable spatio-temporal constraints** (distance + time) to reduce false alarms  
- ✅ **Real-time** pipeline suitable for surveillance deployments

---

## 📊 Results (YOLOv12m baseline vs enhanced training)

Compared to the original YOLOv12m trained under identical conditions:


| Metric | Gain |
|---|---:|
| mAP@0.50 | **+7.1%** |
| mAP@0.50–0.95 | **+7.0%** |
| F1-score | **+7.4%** |

Largest improvements are observed for **small** and **deformable** luggage items.

> Add your detailed class-wise table/plots here if you want:
- `assets/results_table.png`
- `assets/metrics_grid.png`

---

---

# 📦 Dataset Split and Class Distribution

## 🔎 Dataset Overview

The dataset used in this project is **publicly available** and hosted on Roboflow to ensure transparency and reproducibility.

It was constructed from approximately **600 publicly available YouTube videos**, primarily recorded from fixed or quasi-static surveillance viewpoints in:

- Airports
- Railway stations
- Public transport hubs
- Indoor public spaces

The dataset targets **luggage detection** and contains three object classes:

- 🎒 Backpack  
- 👜 Bag  
- 🧳 Trolley  

### 📊 Global Statistics

- **29,053 images**
- **130,475 annotated instances**
- Natural class imbalance preserved intentionally

| Class | Percentage |
|-------|------------|
| Backpack | 26.7% |
| Bag | 21.9% |
| Trolley | 51.3% |

The natural imbalance reflects realistic surveillance distributions and avoids artificial bias during training.

---

## 📂 Dataset Split

| Split | Images (%) | Instances (%) | Backpack (%) | Bag (%) | Trolley (%) |
|--------|------------|---------------|--------------|----------|-------------|
| **Train** | 25,302 (87.1%) | 112,214 (86.0%) | 26.8% | 22.2% | 51.0% |
| **Valid** | 2,954 (10.2%) | 14,371 (11.0%) | 26.9% | 20.1% | 53.0% |
| **Test** | 797 (2.7%) | 3,890 (3.0%) | 24.1% | 22.5% | 53.4% |
| **Total** | 29,053 (100%) | 130,475 (100%) | 26.7% | 21.9% | 51.3% |

The split follows a realistic distribution strategy with a dominant training portion to support robust model learning.

---

# 📏 Object Scale Distribution Analysis

To analyze scale sensitivity, object instances were categorized into **small**, **medium**, and **large** using normalized area thresholds.

Increasing the small-object threshold shifts many instances into the small category, revealing that numerous luggage objects lie near size boundaries.

---

## 🔢 Global Size Distribution

| Small Threshold (S) | Medium Threshold (M) | Small (%) | Medium (%) | Large (%) |
|----------------------|----------------------|-----------|------------|------------|
| 0.0014 (~24×24) | 0.0225 (~96×96) | 12.6% | 71.6% | 15.8% |
| 0.0025 (~32×32) | 0.0225 (~96×96) | 26.4% | 57.8% | 15.8% |
| 0.0039 (~40×40) | 0.0225 (~96×96) | 39.2% | 45.0% | 15.8% |
| 0.0014 (~24×24) | 0.0100 (~64×64) | 12.6% | 54.2% | 33.2% |
| 0.0025 (~32×32) | 0.0100 (~64×64) | 26.4% | 40.4% | 33.2% |

This confirms that the dataset is **scale-sensitive**, motivating small-object–aware training strategies.

---

## 📊 Per-Class Size Distribution Example  
(S = 0.0025, M = 0.0225)

| Class | Total | Small (%) | Medium (%) | Large (%) |
|--------|--------|-----------|------------|------------|
| Backpack | 34,901 | 22.5% | 58.2% | 19.3% |
| Bag | 28,628 | 26.6% | 52.9% | 20.5% |
| Trolley | 66,946 | 28.4% | 59.8% | 11.9% |

Across all configurations, increasing the small threshold consistently increases the proportion of small objects while reducing medium-sized instances.

---

# 🎨 Data Preprocessing and Augmentation

## 🔆 Automatic Contrast Enhancement

Histogram equalization was applied to improve visibility in:

- Low-contrast indoor terminals  
- Poorly illuminated areas  
- Shadow-dominant scenes  

This enhances object boundaries and improves feature extraction reliability.

---

## 🔁 Data Augmentation Strategy

To improve robustness and viewpoint diversity, the following transformations were applied:

- 🔄 **Horizontal flipping** → Left–right invariance  
- 🔁 **Discrete 90° rotations** → Unconventional camera orientations  
- ↩️ **Small-angle rotations (−15° to +15°)** → Natural camera tilt  
- 🔀 **Shearing (±10°)** → Perspective distortions  

Each original training image generated **three additional augmented samples**, resulting in a:

### 🚀 3× Increase in Training Data Volume

This augmentation strategy significantly improves generalization, especially for:

- Small objects  
- Deformable luggage  
- Partially occluded instances  
- Borderline-scale objects  

---

## 🎯 Why This Matters

The dataset exhibits:

- Natural class imbalance  
- High scale sensitivity  
- Frequent small-object presence  
- Real-world surveillance complexity  

These characteristics directly motivate:

- Small-object–aware loss functions  
- Robust augmentation strategies  
- Specialized detection models  

---




