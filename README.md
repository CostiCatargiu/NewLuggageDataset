<!-- =======================================================
🚨 Real-Time Abandoned Luggage Detection (YOLOv12 + Spatio-Temporal)
======================================================= -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/e13235ef-b18c-4dce-9581-bafa0a2a11c4" width="250" />
  <img src="https://github.com/user-attachments/assets/2055d337-f103-4d03-8238-9dbe1b9a0dfe" width="250" />
  <img src="https://github.com/user-attachments/assets/0b9a794e-43f8-4bee-aef4-17c6908df486" width="250" />
</p>

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

Abandoned luggage detection is formulated as a **spatio-temporal reasoning task**, not merely a frame-wise object detection problem.

The system integrates:

- 🧳 **YOLOv12m (custom-trained)** — specialized for luggage detection (backpack, bag, trolley) with enhanced small-object sensitivity.
- 🧍 **YOLOv12x** — high-recall person detector optimized for crowded surveillance environments.

Instead of relying on appearance-based trackers (e.g., DeepSORT), a **custom motion-based tracking-by-detection framework** is employed, combining:

- IoU-based geometric matching  
- Distance gating  
- Constant-velocity motion prediction  
- Track smoothing for temporal stability  

### 🎯 Abandonment Logic

A luggage track ℓ is classified as *abandoned* when:

- No person track p satisfies  
  `||c_ℓ − c_p||₂ ≤ R`
- For a continuous duration  
  `u(t) ≥ T_unattended`

This explicit formulation improves interpretability and reduces false alarms caused by brief separations or detector fluctuations.

---

# 🖼 Example Dataset Samples

<p align="center">
<img width="924" src="https://github.com/user-attachments/assets/6b436ba9-0d92-4bc3-b14d-9e35fa155664">
</p>

### 📸 Dataset Characteristics

The dataset was constructed from approximately **600 publicly available YouTube surveillance videos**, primarily recorded from fixed or quasi-static viewpoints in:

- Airports  
- Railway stations  
- Indoor public areas  

It captures realistic surveillance conditions, including:

- Frequent occlusions and crowd interactions  
- Illumination variability (indoor/outdoor, glare, shadows)  
- Small and medium-scale luggage instances  
- Natural class imbalance  
- Compression artifacts and motion blur  

These properties closely reflect real-world deployment environments, making the dataset highly representative for practical public safety systems.

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

The class distribution is naturally imbalanced:

- **Trolley ≈ 51%**
- **Backpack ≈ 27%**
- **Bag ≈ 22%**

The imbalance was intentionally preserved to reflect real surveillance statistics rather than artificially balancing the dataset.

---

# 📂 Dataset Split

| Split | Images (%) | Instances (%) | Backpack (%) | Bag (%) | Trolley (%) |
|--------|------------|---------------|--------------|----------|-------------|
| **Train** | 25,302 (87.1%) | 112,214 (86.0%) | 26.8% | 22.2% | 51.0% |
| **Valid** | 2,954 (10.2%) | 14,371 (11.0%) | 26.9% | 20.1% | 53.0% |
| **Test** | 797 (2.7%) | 3,890 (3.0%) | 24.1% | 22.5% | 53.4% |
| **Total** | 29,053 (100%) | 130,475 (100%) | 26.7% | 21.9% | 51.3% |

The training-heavy split supports stable optimization while preserving representative validation and test distributions.

---

# 📏 Object Scale Distribution

To quantify scale sensitivity, object instances were categorized using normalized area thresholds.

A significant proportion of luggage objects lie near small-object boundaries, motivating scale-aware training modifications.

## 🔹 S = 0.001400 (~24×24) | M = 0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 16,476 (12.6%) | 93,435 (71.6%) | 20,564 (15.8%) |
| backpack | 34,901 | 3,490 (10.0%) | 24,662 (70.7%) | 6,749 (19.3%) |
| bag | 28,628 | 3,632 (12.7%) | 19,136 (66.8%) | 5,860 (20.5%) |
| trolley | 66,946 | 9,354 (14.0%) | 49,637 (74.1%) | 7,955 (11.9%) |

Increasing the small-object threshold to ~40×40 pixels raises the small-instance proportion to **39.2%**, confirming strong scale sensitivity within the dataset.

---

# 🎨 Preprocessing & Augmentation (Roboflow Pipeline)

All preprocessing and augmentation operations were performed **offline on the Roboflow platform prior to training**.  
The transformed images were exported as part of the final dataset and used directly during model optimization.

## 🔹 Preprocessing

- Auto-orient (EXIF-based correction)
- Resize to **640×640**
- Adaptive histogram equalization (contrast enhancement)

These steps standardize scale, prevent orientation bias, and improve boundary visibility for small objects.

---

## 🔹 Augmentation (Roboflow Dataset Expansion)

Each training image generated **three independent augmented variants (3× expansion)** using Roboflow’s stochastic augmentation engine.

Applied transformations:

- Horizontal flip  
- Rotation (−14° to +14°)  
- Shear (±13°)  
- Grayscale conversion (10% probability)  
- Gaussian blur (≤ 1.6 px kernel)

### 🔎 Why This Matters

Surveillance imagery commonly exhibits:

- Lighting variation  
- Motion blur  
- Perspective distortion  
- Reduced color fidelity  
- Small-object localization difficulty  

The Roboflow-based augmentation pipeline increases appearance diversity while maintaining annotation consistency, leading to:

- Improved robustness to illumination changes  
- Better generalization to unseen camera viewpoints  
- Increased stability for small-object detection  
- Reduced overfitting to specific scene conditions  

---

# 📈 Performance Improvements

| Metric | Improvement |
|---|---|
| mAP@0.50 | +7.1% |
| mAP@0.50–0.95 | +7.0% |
| F1-score | +7.4% |

The strongest gains are observed for small and medium-scale luggage instances, validating the proposed scale-aware training modification.

---
