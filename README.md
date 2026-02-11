<!-- =======================================================
🚨 Real-Time Abandoned Luggage Detection (YOLOv12 + Spatio-Temporal)
======================================================= -->

<div align="center">

# 🚨 Real-Time Abandoned Luggage Detection  
## Enhanced YOLOv12 + Tracking-by-Detection + Spatio-Temporal Reasoning

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" />
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLOv12-111827?logo=github&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p>
A real-time abandoned luggage detection framework integrating dual YOLOv12 models, motion-based tracking-by-detection, and explicit spatio-temporal reasoning for reliable public safety surveillance.
</p>

</div>

---

# 🧠 System Architecture

<p align="center">
<img width="761" src="https://github.com/user-attachments/assets/beec2c70-dd65-403a-90a1-1b331a8028ab">
</p>

### 🔍 Architecture Overview

The proposed system formulates abandoned luggage detection as a **spatio-temporal reasoning problem**, not merely a frame-wise detection task.

The architecture consists of:

- 🧳 **YOLOv12m (Custom-trained)** — optimized for luggage detection (backpack, bag, trolley) with improved small-object sensitivity.
- 🧍 **YOLOv12x** — high-recall person detector for crowded surveillance environments.

Rather than relying on appearance-based trackers (e.g., DeepSORT), the system uses a **custom motion-based tracking-by-detection strategy**:

- Geometric matching via IoU and distance constraints
- Constant-velocity motion prediction
- Track-level smoothing for stability

### 🎯 Abandonment Logic

A luggage track ℓ is considered *abandoned* if:

- No person track p satisfies  
  `||c_ℓ − c_p||₂ ≤ R`
- For a continuous duration  
  `u(t) ≥ T_unattended`

This interpretable formulation ensures robust behavior under occlusions, short-term separation, and detection noise.

---

# 🖼 Example Dataset Samples

<p align="center">
<img width="924" src="https://github.com/user-attachments/assets/6b436ba9-0d92-4bc3-b14d-9e35fa155664">
</p>

### 📸 Dataset Characteristics

The dataset contains surveillance-style frames extracted from approximately **600 publicly available YouTube videos**, primarily recorded from fixed or quasi-static viewpoints in:

- Airports  
- Railway stations  
- Indoor public areas  

Key challenges include:

- Frequent occlusions  
- Illumination variability  
- Small-scale luggage instances  
- Natural class imbalance  
- Realistic crowd density  

These properties closely match real-world deployment conditions.

---

# 📊 Dataset Summary (Official Statistics)

## 🔎 General Information

| Attribute | Value |
|------------|--------|
| 📸 Images | **29,053** |
| 🏷️ Instances | **130,475** |
| 🧳 Classes | backpack, bag, trolley |
| 📦 Format | YOLO (normalized coordinates) |
| 📜 License | MIT |
| 🌍 Hosting | https://universe.roboflow.com/guns-detection-cvwjs/luggagedataset-24pgo |
| 📊 Model Results | https://drive.google.com/drive/folders/12aaS7CwZfGqb7__BK1UX54j1gQS_DoPi |

The natural class distribution (≈51% trolley, ≈27% backpack, ≈22% bag) was preserved to avoid artificial balancing bias.

---

# 📂 Dataset Split

| Split | Images (%) | Instances (%) | Backpack (%) | Bag (%) | Trolley (%) |
|--------|------------|---------------|--------------|----------|-------------|
| **Train** | 25,302 (87.1%) | 112,214 (86.0%) | 26.8% | 22.2% | 51.0% |
| **Valid** | 2,954 (10.2%) | 14,371 (11.0%) | 26.9% | 20.1% | 53.0% |
| **Test** | 797 (2.7%) | 3,890 (3.0%) | 24.1% | 22.5% | 53.4% |
| **Total** | 29,053 (100%) | 130,475 (100%) | 26.7% | 21.9% | 51.3% |

The training-heavy distribution supports robust optimization while maintaining realistic validation and testing subsets.

---

# 📏 Object Scale Distribution

To quantify scale sensitivity, instances were categorized using normalized area thresholds.

Many objects lie near small-object boundaries, motivating scale-aware training.

## 🔹 S = 0.001400 (~24×24) | M = 0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 16,476 (12.6%) | 93,435 (71.6%) | 20,564 (15.8%) |
| backpack | 34,901 | 3,490 (10.0%) | 24,662 (70.7%) | 6,749 (19.3%) |
| bag | 28,628 | 3,632 (12.7%) | 19,136 (66.8%) | 5,860 (20.5%) |
| trolley | 66,946 | 9,354 (14.0%) | 49,637 (74.1%) | 7,955 (11.9%) |

Increasing the small threshold to ~40×40 pixels raises the small-instance proportion to 39.2%, confirming the dataset’s scale-sensitive nature.

---

# 🎨 Preprocessing & Augmentation (Roboflow Pipeline)

All preprocessing and augmentation operations were performed **offline on the Roboflow platform prior to training**.

## 🔹 Preprocessing

- Auto-orient (EXIF-based correction)
- Resize to **640×640**
- Adaptive histogram equalization (contrast enhancement)

## 🔹 Augmentation (Roboflow Dataset Expansion)

Each training image generated **three independent augmented variants (3× expansion)**.

Stochastic augmentations included:

- Horizontal flip
- Rotation (−14° to +14°)
- Shear (±13°)
- Grayscale conversion (10% probability)
- Gaussian blur (≤ 1.6 px kernel)

Because augmentation was applied during dataset generation, the training set was fixed prior to model optimization.

This strategy improves robustness to:

- Illumination changes
- Motion blur
- Scale variation
- Perspective distortion
- Small-object localization challenges

---

# 📈 Performance Improvements

| Metric | Improvement |
|---|---|
| mAP@0.50 | +7.1% |
| mAP@0.50–0.95 | +7.0% |
| F1-score | +7.4% |

The strongest gains are observed for small and medium-scale luggage instances, validating the proposed scale-aware training modification.

---
