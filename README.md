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

# 🧠 System Overview

<p align="center">
<img width="761" src="https://github.com/user-attachments/assets/beec2c70-dd65-403a-90a1-1b331a8028ab">
</p>

# 🖼 Example Dataset Samples

<p align="center">
<img width="924" src="https://github.com/user-attachments/assets/6b436ba9-0d92-4bc3-b14d-9e35fa155664">
</p>


---
The system integrates:

- 🧳 YOLOv12m → luggage detection  
- 🧍 YOLOv12x → person detection  
- 🔄 Motion-based tracking-by-detection  
- 📏 Distance-based supervision radius  
- ⏱ Duration-based unattended logic  

A luggage item is flagged as abandoned only if it remains outside supervision radius **R** for at least **T_unattended** seconds.

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

---

# 📂 Dataset Split

| Split | Images (%) | Instances (%) | Backpack (%) | Bag (%) | Trolley (%) |
|--------|------------|---------------|--------------|----------|-------------|
| **Train** | 25,302 (87.1%) | 112,214 (86.0%) | 26.8% | 22.2% | 51.0% |
| **Valid** | 2,954 (10.2%) | 14,371 (11.0%) | 26.9% | 20.1% | 53.0% |
| **Test** | 797 (2.7%) | 3,890 (3.0%) | 24.1% | 22.5% | 53.4% |
| **Total** | 29,053 (100%) | 130,475 (100%) | 26.7% | 21.9% | 51.3% |

---

# 📏 Object Scale Distribution

The dataset was analyzed under multiple normalized area thresholds.

## 🔹 S=0.001400 (~24×24) | M=0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 16,476 (12.6%) | 93,435 (71.6%) | 20,564 (15.8%) |
| backpack | 34,901 | 3,490 (10.0%) | 24,662 (70.7%) | 6,749 (19.3%) |
| bag | 28,628 | 3,632 (12.7%) | 19,136 (66.8%) | 5,860 (20.5%) |
| trolley | 66,946 | 9,354 (14.0%) | 49,637 (74.1%) | 7,955 (11.9%) |

---

## 🔹 S=0.002500 (~32×32) | M=0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 34,449 (26.4%) | 75,462 (57.8%) | 20,564 (15.8%) |
| backpack | 34,901 | 7,846 (22.5%) | 20,306 (58.2%) | 6,749 (19.3%) |
| bag | 28,628 | 7,617 (26.6%) | 15,151 (52.9%) | 5,860 (20.5%) |
| trolley | 66,946 | 18,986 (28.4%) | 40,005 (59.8%) | 7,955 (11.9%) |

---

## 🔹 S=0.003900 (~40×40) | M=0.022500 (~96×96)

| Group | Total | Small | Medium | Large |
|--------|--------|--------|--------|--------|
| TOTAL | 130,475 | 51,149 (39.2%) | 58,762 (45.0%) | 20,564 (15.8%) |
| backpack | 34,901 | 12,104 (34.7%) | 16,048 (46.0%) | 6,749 (19.3%) |
| bag | 28,628 | 11,266 (39.4%) | 11,502 (40.2%) | 5,860 (20.5%) |
| trolley | 66,946 | 27,779 (41.5%) | 31,212 (46.6%) | 7,955 (11.9%) |

---



# 🎨 Data Augmentation

Applied during training:

- Horizontal flipping  
- 90° rotations  
- Small-angle rotations (−15° to +15°)  
- Shearing (±10°)  
- Histogram equalization  

Each image generated **3 augmented variants**, resulting in **3× training expansion**.

---

# 📈 Performance Gains

| Metric | Improvement |
|---|---|
| mAP@0.50 | +7.1% |
| mAP@0.50–0.95 | +7.0% |
| F1-score | +7.4% |

Largest improvements observed for small and deformable luggage instances.

---
