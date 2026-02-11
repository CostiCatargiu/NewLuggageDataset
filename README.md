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
Real-time abandoned luggage detection framework using enhanced YOLOv12 models and explicit spatio-temporal reasoning. A custom-trained YOLOv12m improves small-object detection, while YOLOv12x ensures robust person tracking.
</p>

</div>

---

# 🧠 System Architecture

<p align="center">
<img width="761" src="https://github.com/user-attachments/assets/beec2c70-dd65-403a-90a1-1b331a8028ab">
</p>

### 🔍 Architecture Description

The proposed system integrates **dual deep-learning detectors** with explicit temporal reasoning:

- 🧳 **YOLOv12m** — specialized for luggage detection (backpack, bag, trolley)  
- 🧍 **YOLOv12x** — optimized for robust person detection in crowded scenes  

Instead of relying on off-the-shelf multi-object trackers, the system employs a **motion-based tracking-by-detection strategy**. Bounding boxes are associated across frames using geometric consistency and motion prediction, ensuring stable object identities.

Abandonment is determined through:

- 📏 A supervision radius **R**
- ⏱ A minimum unattended duration **T_unattended**

This explicit formulation ensures interpretability and reduces false alarms caused by brief separations or detection noise.

---

# 🖼 Example Dataset Samples

<p align="center">
<img width="924" src="https://github.com/user-attachments/assets/6b436ba9-0d92-4bc3-b14d-9e35fa155664">
</p>

### 📸 Sample Description

The dataset consists of surveillance-style frames captured from real-world environments such as airports, train stations, and indoor terminals.  

Key characteristics include:

- Frequent occlusions
- Varying illumination conditions
- Small and medium-sized luggage instances
- Natural class imbalance
- Realistic crowd density

These properties make the dataset highly representative of practical surveillance deployments.

---

# 📊 Dataset Summary (Official Statistics)

## 🔎 General Information

| Attribute | Value |
|------------|--------|
| 📸 Images | **29,053** |
| 🏷️ Instances | **130,475** |
| 🧳 Classes | backpack, bag, trolley |
| 📦 Format | YOLO (normalized x_center, y_center, width, height) |
| 📜 License | MIT |
| 🌍 Hosting | https://universe.roboflow.com/guns-detection-cvwjs/luggagedataset-24pgo |
| 📊 Training Results | https://drive.google.com/drive/folders/12aaS7CwZfGqb7__BK1UX54j1gQS_DoPi |

### 📖 Description

The dataset was curated from approximately **600 publicly available YouTube videos**, primarily recorded from fixed or quasi-static surveillance viewpoints.

The natural class imbalance (≈51% trolleys, ≈27% backpacks, ≈22% bags) was intentionally preserved to reflect real-world distribution patterns.

---

# 📂 Dataset Split

| Split | Images (%) | Instances (%) | Backpack (%) | Bag (%) | Trolley (%) |
|--------|------------|---------------|--------------|----------|-------------|
| **Train** | 25,302 (87.1%) | 112,214 (86.0%) | 26.8% | 22.2% | 51.0% |
| **Valid** | 2,954 (10.2%) | 14,371 (11.0%) | 26.9% | 20.1% | 53.0% |
| **Test** | 797 (2.7%) | 3,890 (3.0%) | 24.1% | 22.5% | 53.4% |
| **Total** | 29,053 (100%) | 130,475 (100%) | 26.7% | 21.9% | 51.3% |

### 📖 Split Description

The dataset was divided following a realistic training-heavy strategy:

- 87.1% training data
- 10.2% validation data
- 2.7% testing data

This ensures sufficient data for robust model optimization while maintaining representative evaluation subsets.

---

# 📏 Object Scale Distribution

The dataset was analyzed under multiple normalized area thresholds to quantify scale sensitivity.

Many luggage instances lie near small-object boundaries, motivating the introduction of a **small-object–aware loss function**.

---

## 🔹 S=0.001400 (~24×24) | M=0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 16,476 (12.6%) | 93,435 (71.6%) | 20,564 (15.8%) |
| backpack | 34,901 | 3,490 (10.0%) | 24,662 (70.7%) | 6,749 (19.3%) |
| bag | 28,628 | 3,632 (12.7%) | 19,136 (66.8%) | 5,860 (20.5%) |
| trolley | 66,946 | 9,354 (14.0%) | 49,637 (74.1%) | 7,955 (11.9%) |

### 📖 Interpretation

Under the strictest small-object threshold (~24×24 pixels), only 12.6% of instances are categorized as small.  
However, as the threshold increases, the proportion of small objects rises significantly, confirming the dataset’s scale-sensitive nature.

---

# 🎨 Data Augmentation Strategy

Applied during training:

- Horizontal flipping  
- 90° rotations  
- Small-angle rotations (−15° to +15°)  
- Shearing (±10°)  
- Histogram equalization  

### 📖 Augmentation Description

To enhance generalization and simulate real surveillance variability:

- Horizontal flipping introduces viewpoint invariance.
- Rotations simulate unconventional camera orientations.
- Small-angle rotations model camera tilt.
- Shearing reproduces perspective distortions.
- Histogram equalization improves visibility in low-light scenes.

Each image generated **three augmented variants**, effectively producing a **3× increase in training data volume**.

This significantly improves robustness for:

- Small objects
- Deformable luggage
- Partial occlusions
- Borderline-scale instances

---

# 📈 Performance Gains

| Metric | Improvement |
|---|---|
| mAP@0.50 | +7.1% |
| mAP@0.50–0.95 | +7.0% |
| F1-score | +7.4% |

The strongest improvements are observed for small and medium-scale luggage items, validating the scale-aware training strategy.

---
