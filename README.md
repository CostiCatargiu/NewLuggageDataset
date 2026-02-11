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

## ✨ Key contributions

- ✅ **Large-scale public dataset**: 29,053 images / 130,475 instances (bag, backpack, trolley)  
- ✅ **Small-object–aware loss (training-only change)** — YOLOv12m architecture unchanged  
- ✅ **Robust pairing** of person ↔ luggage via tracking-by-detection  
- ✅ **Interpretable spatio-temporal constraints** (distance + time) to reduce false alarms  
- ✅ **Real-time** pipeline suitable for surveillance deployments

---

## 📊 Results (YOLOv12m baseline vs enhanced training)

Compared to the original YOLOv12m trained under identical conditions:

D_t^P = {(b_i^P, c_i^P)}
D_t^L = {(b_j^L, k_j^L, c_j^L)}


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

## 🧠 Pipeline overview

```text
Video → Person Detector (YOLOv12x) ┐
                                   ├→ Tracking-by-Detection → Pairing
Video → Luggage Detector (YOLOv12m)┘
                          ↓
              Spatio-temporal logic:
  "outside supervision radius R for >= T seconds" → ABANDONED
