<!-- =======================================================
🚨 Real-Time Abandoned Luggage Detection (YOLOv12 + Spatio-Temporal)
======================================================= -->

<div align="center">

# 🚨 Real-Time Abandoned Luggage Detection  
## Dual YOLOv12 Models + Tracking-by-Detection + Spatio-Temporal Reasoning

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" />
  <img alt="Ultralytics" src="https://img.shields.io/badge/Ultralytics-YOLOv12-111827?logo=github&logoColor=white" />
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green" />
</p>

<p>
A real-time surveillance framework that detects and classifies unattended luggage using specialized object detection models and interpretable spatio-temporal constraints.
</p>

</div>

---

# 🧠 System Architecture

<p align="center">
<img width="761" src="https://github.com/user-attachments/assets/beec2c70-dd65-403a-90a1-1b331a8028ab">
</p>

### 🔍 Overview

Abandoned luggage detection is formulated as a **spatio-temporal reasoning task**, not simply frame-level object detection.

The framework integrates:

- 🧳 **YOLOv12m (custom-trained)** — optimized for small and deformable luggage (backpack, bag, trolley)
- 🧍 **YOLOv12x** — high-recall person detection for crowded scenes
- 🔄 **Custom tracking-by-detection** — motion-based geometric association
- 📏 **Distance-based supervision constraint**
- ⏱ **Duration-based abandonment logic**

Unlike appearance-based trackers (e.g., DeepSORT), the system relies on:

- IoU-based matching
- Distance gating
- Constant-velocity prediction
- Track smoothing

This ensures stable identities while maintaining computational efficiency.

### 🎯 Abandonment Condition

A luggage track ℓ is declared **abandoned** if:

- No person p satisfies  
  `||c_ℓ − c_p||₂ ≤ R`
- For a continuous duration  
  `u(t) ≥ T_unattended`

This interpretable formulation prevents false alarms from brief separations or detector noise.

---

# 🖼 Example Dataset Samples

<p align="center">
<img width="924" src="https://github.com/user-attachments/assets/6b436ba9-0d92-4bc3-b14d-9e35fa155664">
</p>

### 📸 Dataset Characteristics

The dataset was constructed from approximately **600 publicly available YouTube surveillance videos**, primarily recorded from fixed or quasi-static viewpoints.

It reflects realistic operational conditions, including:

- 👥 Crowd density variation  
- 🌗 Illumination changes  
- 🧳 Small and medium-scale luggage  
- 🔁 Partial occlusions  
- ⚖ Natural class imbalance  
- 📷 Compression artifacts and motion blur  

These characteristics make it representative of real-world public safety deployments.

---

# 📊 Dataset Summary

## 🔎 General Information

| Attribute | Value |
|------------|--------|
| 📸 Images | **29,053** |
| 🏷️ Instances | **130,475** |
| 🧳 Classes | backpack, bag, trolley |
| 📦 Format | YOLO (normalized coordinates) |
| 📜 License | MIT |
| 🌍 Hosting | Roboflow Universe |
| 📊 Model Results | Google Drive (link above) |

The class distribution is naturally imbalanced:

- **Trolley** ≈ 51%  
- **Backpack** ≈ 27%  
- **Bag** ≈ 22%  

The imbalance was intentionally preserved to reflect real surveillance data.

---

# 📂 Dataset Split

| Split | Images (%) | Instances (%) | Backpack (%) | Bag (%) | Trolley (%) |
|--------|------------|---------------|--------------|----------|-------------|
| **Train** | 25,302 (87.1%) | 112,214 (86.0%) | 26.8% | 22.2% | 51.0% |
| **Valid** | 2,954 (10.2%) | 14,371 (11.0%) | 26.9% | 20.1% | 53.0% |
| **Test** | 797 (2.7%) | 3,890 (3.0%) | 24.1% | 22.5% | 53.4% |
| **Total** | 29,053 (100%) | 130,475 (100%) | 26.7% | 21.9% | 51.3% |

The training-heavy split ensures strong optimization while maintaining representative validation and test subsets.

---

# 📏 Object Scale Distribution

To analyze scale sensitivity, instances were categorized using normalized area thresholds.

Many luggage objects lie near small-object boundaries, motivating the use of scale-aware training modifications.

## 🔹 S = 0.001400 (~24×24) | M = 0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 16,476 (12.6%) | 93,435 (71.6%) | 20,564 (15.8%) |
| backpack | 34,901 | 3,490 (10.0%) | 24,662 (70.7%) | 6,749 (19.3%) |
| bag | 28,628 | 3,632 (12.7%) | 19,136 (66.8%) | 5,860 (20.5%) |
| trolley | 66,946 | 9,354 (14.0%) | 49,637 (74.1%) | 7,955 (11.9%) |

Raising the small-object threshold increases small-instance proportion to **39.2%**, confirming strong scale sensitivity within the dataset.

---

# 🎨 Preprocessing & Augmentation (Roboflow Platform)

All preprocessing and augmentation operations were performed **offline on the Roboflow platform prior to training**.  
The augmented images were exported as part of the final fixed training dataset.

## 🔹 Preprocessing

- Auto-orient (EXIF-based correction)
- Resize to **640×640**
- Adaptive histogram equalization (contrast enhancement)

These steps standardize scale, prevent rotational bias, and improve boundary visibility for small objects.

---

## 🔹 Dataset Expansion (3× Augmentation)

Each training image generated **three independent augmented variants**.

Stochastic augmentations included:

- Horizontal flip  
- Rotation (−14° to +14°)  
- Shear (±13°)  
- Grayscale conversion (10% probability)  
- Gaussian blur (≤1.6 px kernel)

### 🔎 Why This Matters

Surveillance imagery often suffers from:

- Motion blur  
- Perspective distortion  
- Lighting variability  
- Camera misalignment  
- Reduced color fidelity  

The Roboflow-based augmentation pipeline increases appearance diversity while preserving label integrity.

This leads to:

- Improved small-object localization
- Greater illumination robustness
- Reduced overfitting
- Stronger generalization to unseen surveillance footage

---

# 📈 Performance Improvements

| Metric | Improvement |
|---|---|
| mAP@0.50 | +7.1% |
| mAP@0.50–0.95 | +7.0% |
| F1-score | +7.4% |

Performance gains are most pronounced for small and medium-scale luggage instances, validating the scale-aware training strategy.

---
