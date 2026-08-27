# Unattended / Abandoned Luggage Detection — Deep Literature Review

**Scope:** all 32 distinct papers in `RelatedPaper/` (three of the PDFs were byte-identical duplicates, and the 115-page `Computer_Vision_and_Pattern_RecognitionCVPR06_In_.pdf` is the **full PETS 2006 workshop proceedings**, which contains 8 separate left-luggage papers — all of them are broken out individually below).

**Focus (as requested):** *public datasets* and *methodology*, per paper.

**Compiled:** 22 Aug 2026

---

## 0. How to read this document

| Section | What's in it |
|---|---|
| **§1** | **Public dataset catalogue** — every benchmark used across the corpus, with size, content, annotations, availability, and which papers used it. This is the section to mine for your own dataset decisions. |
| **§2** | The problem definition itself (the PETS 2006 rule set that ~80% of this literature still uses) |
| **§3** | Per-paper deep dives, grouped into 5 methodological eras |
| **§4** | Master comparison table (paper × dataset × method × metrics) |
| **§5** | Methodology taxonomy — the recurring algorithmic building blocks |
| **§6** | Cross-cutting synthesis: what actually works, what's broken, open gaps, and caveats about reported numbers |

**Notation:** ✅ public & findable · ⚠️ public but degraded/hard to obtain · 🔒 private/custom

### Coverage statement (audited)

All 28 unique PDFs were converted to text and read; the 115-page proceedings volume was split into its 8 constituent papers first. **Total: 32 distinct papers.** Extraction was verified — no PDF produced empty or image-only text, so nothing needed OCR.

This document was then **re-audited line-by-line against the extracted text**, because the first pass had read 8 papers section-targeted rather than end-to-end. That second pass found and fixed the following, all now incorporated:

| Paper | What was missing or wrong on the first pass |
|---|---|
| **Zhou & Xu 2024** | The real ablation (Table 5) and the 9-model comparison (Table 6) were absent — including the key fact that **SAO-YOLO is 4× *smaller* than its baseline (7.0 M → 1.7 M params)** and beats TPH-YOLOv5 with 13× fewer parameters. Also missing: the adaptive learning-rate equations, the 4 pixel states / 5 FSM states / 2 timers of the improved PFSM, the SAO-FPN layer surgery, and the LFEM Q/K/V mechanics. |
| **Vrsalovic 2025 (Sensors)** | Their CCTV-Korzo / Düsseldorf / KD datasets were wrongly marked 🔒 **private — they are publicly released** on GitHub and Roboflow. Corrected, and promoted to a full entry in §1.2. |
| **Chen et al. 2024** | The ROI Approach-1-vs-2 numbers (62.4 % vs 51.8 % area ratio; 52.3 ms vs 44.6 ms) and the ray-casting odd-intersection rule were missing. The metric-transposition flag is now resolved (abstract + conclusion agree; §5 is the outlier). |
| **Chang et al. 2010** | The alarm table was reproduced with the wrong column labels and one column omitted. Also missing: the explicit `w(i,j)` formula, the **Δt = 30 s vs the 2008 paper's 60 s** change, the no-dynamic-background limitation, multi-scale templates, and the χ² histogram distance. |
| **Otoom et al. 2008** | Missing the concrete **">2/3 occlusion breaks the classifier"** threshold, the per-frame outcomes, the authors' own confound admission, and their generalisation caveat. |
| **Song et al. 2025** | Missing the Inner-IoU / MPDIoU formulation and the fact that their data is **not publicly available** (third-party restriction). |

**Duplicate handling.** Four byte-identical files were skipped. Two further *content* duplicates were verified by comparing every headline figure across both files and confirmed identical: `1803.01160.pdf` ≡ `Real-Time_Deep_Learning_...pdf` (Smeureanu & Ionescu), and `Human-Object_Interaction_...pdf` ≡ `Dogariu_COMM_2020.pdf`. These are counted once each.

**Numbers verification.** Every headline metric quoted here was re-checked by string-matching it back against the extracted source text.

---

## 1. Public dataset catalogue

This is the single most important table in the review. Read it before designing experiments — the field's evaluation practice is fragmented and several "standard" benchmarks are effectively dead.

### 1.1 The four core benchmarks

#### **PETS 2006** ⚠️ — the foundational left-luggage benchmark

| Property | Value |
|---|---|
| **Full name** | 9th IEEE Int'l Workshop on Performance Evaluation of Tracking and Surveillance, 2006 dataset |
| **Origin paper** | Thirde, Li & Ferryman, *"Overview of the PETS2006 Challenge"*, PETS 2006, pp. 47–50 |
| **Location** | **Victoria Station, London, UK** (real working railway station), filmed with the support of British Transport Police and Network Rail |
| **Size** | **7 multi-camera sequences** (S1–S7), each ~94–136 s |
| **Cameras** | **4 synchronised DV cameras** — 2× Canon MV-1 (1×CCD progressive scan), 2× Sony DCR-PC1000E (3×CMOS) |
| **Resolution / rate** | **768 × 576 (PAL), 25 fps**, distributed as JPEG image sequences (~90 % quality) |
| **Calibration** | **Tsai camera model**, computed from geometric floor patterns; supplied as an XML file per scenario |
| **Ground truth** | Second XML per scenario: radii *a*/*b*, luggage location, warning + alarm trigger frames, actor/luggage counts |
| **Luggage types (5)** | briefcase, suitcase, 25 L rucksack, 75 L backpack (some papers say 70 L), ski gear carrier |
| **Constraint** | Each luggage has exactly one owner; each person owns at most one luggage |
| **Training data** | **NONE** — test sequences only. Smith et al. explicitly flag this as a shortcoming and admit unavoidable tuning on the test set. |
| **Legal** | UK Information Commissioner approved public release for academic research; video © ISCAPS consortium; EU grant SEC4-PR-013800 |
| **Availability 2026** | Original `cvg.reading.ac.uk/PETS2006/data.html` is dead. Vrsalovic et al. (PICom 2025) cite a **web.archive.org snapshot** and state PETS2006/AVSS2007 "are small and have become difficult to obtain and are **no longer available**." Shah et al. (2026) cite `ftp://ftp.cs.rdg.ac.uk/pub/PETS2006/`. |

**The seven scenarios (canonical descriptions from the overview paper + PETS difficulty ratings ★):**

| Seq | Take | Content | Luggage | Difficulty |
|---|---|---|---|---|
| S1 | 1-C | One person with a rucksack loiters, then abandons it | 1 backpack | ★☆☆☆☆ |
| S2 | 3-C | Two people enter from opposite directions; one places a suitcase; both leave without it | 1 suitcase | ★★★☆☆ |
| S3 | 7-A | Person waiting for a train temporarily places a briefcase, then **picks it up again** — *negative control, no alarm* | 1 briefcase | ★☆☆☆☆ |
| S4 | 5-A | Person places suitcase; 2nd person arrives and talks; 1st leaves without luggage; 2nd (reading a newspaper) doesn't notice | 1 suitcase | ★★★★☆ |
| S5 | 1-G | Single person with **ski equipment** loiters then abandons it | ski gear carrier | ★★☆☆☆ |
| S6 | 3-H | Two people enter together; one places a rucksack; both leave without it | 1 rucksack | ★★★☆☆ |
| S7 | 6-B | Person with suitcase loiters, abandons it; **five other people move in close proximity** | 1 suitcase | ★★★★★ |

> ⚠️ **Fact-check flag:** Shah et al. (IEEE Access 2026) describe PETS 2006 as *"recorded in a parking lot in an outdoor location"*. This is **wrong** — it is an indoor railway station concourse. Treat that paper's dataset description with caution.

---

#### **i-LIDS AVSS 2007 (AB — Abandoned Baggage)** ⚠️

| Property | Value |
|---|---|
| **Origin** | UK Home Office *Imagery Library for Intelligent Detection Systems*, released for AVSS 2007 |
| **Content** | **3 single-view video sequences** of a **London Underground / metro platform**, one per difficulty tier: **AB-Easy, AB-Medium, AB-Hard** |
| **Difficulty axis** | Distance of the drop-off zone from the camera, object size in pixels, and amount of crowd occlusion |
| **Resolution** | 720 × 576, 25 fps (D1) |
| **Events** | One abandonment event per sequence |
| **Known trap** | In Medium and Hard, the owner **passes behind a large pillar (~1.5 s ≈ 40 frames occlusion)** before leaving — this is exactly where tracking-based methods break (Liao 2008, Chang 2010 both lose the owner here) |
| **Availability** | `eecs.qmul.ac.uk/~andrea/avss2007_d.html` — historically only 3 of 8 clips released free; Chang et al. note *"i-LIDS did not provide all test cases for free."* Now largely unavailable. |
| **Alarm rule** | Uses **T_L = 60 s / owner-left-scene** convention (differs from PETS 2006's 30 s) — Chang et al. explicitly note this discrepancy |

---

#### **PETS 2007** ✅ (partially)

| Property | Value |
|---|---|
| **Content** | Airport-style scenarios: **loitering, luggage theft, and abandoned/re-attended baggage** |
| **Relevant subsets** | **S7** — person with two bags, leaves one accidentally, then returns for it. **S8** — person places a large bag, walks away, then retrieves it. |
| **Cameras** | 4 views; camera 1 has the hardest geometry (worst angle, strongest illumination change) — Dahi et al. deliberately use it as a stress test |
| **Luggage types** | handbag, carry-on case, 70 L backpack, ski gear carrier |
| **Known weakness** | Only **2 abandonment scenarios**, insufficient for training; Otoom et al. had to borrow objects from S1–S8 to build a 4-class classification set |
| **Used by** | Dahi 2017, Smeureanu & Ionescu 2018, Otoom 2008, Beleznai 2013 (cited) |

---

#### **ABODA** ✅ — the de-facto modern benchmark

| Property | Value |
|---|---|
| **Full name** | Abandoned Objects DAtaset |
| **Origin** | **Lin, Chen, Chen, Lin & Hung**, *"Abandoned Object Detection via Temporal Consistency Modeling and Back-Tracing Verification"*, IEEE TIFS 10(7):1359–1370, 2015 |
| **URL** | **`https://github.com/kevinlin311tw/ABODA`** ✅ (open source, still live) |
| **Size** | **11 video sequences**, 12 ground-truth abandonment events total (video 6 has 2) |
| **Coverage** | indoor · outdoor · **night-time / IR** · **light-switching (complete background change)** · **crowded** |
| **⚠️ Critical limitation** | **Unannotated.** Ngo & Mutaher state plainly: *"this dataset remains unannotated, and dedicating time to label the data is not a viable option."* Smeureanu & Ionescu had to **manually annotate bounding boxes themselves**. Every paper therefore evaluates on its own labels → **the reported numbers are not strictly comparable.** |
| **Hard cases** | **Video 5** (night vision — bags read as glare or blend into background), **Video 6** (2 objects), **Videos 7–8** (light switching), **Video 11** (crowded + very small object — *almost every method fails here*: HLDNet ✗, Dahi ✗, Soontornnapar ✗) |
| **Used by** | Lin 2015 (origin), Dahi 2017, Altunay 2018, Kim/HLDNet 2022, Zhou & Xu 2024, Soontornnapar 2025, Ngo & Mutaher 2025, Shah 2026 |

**Cross-paper ABODA scoreboard** (as reported by each paper — read with the annotation caveat above):

| Method | Precision | Recall | F1 | Note |
|---|---|---|---|---|
| Park et al. 2019 (dual bg + illum.) | 100 % | 100 % | 100 % | as re-reported by Soontornnapar |
| Newlin/Russel & Selvaraj 2024 | 91.67 % | 100 % | 95.65 % | as re-reported |
| **Zhou & Xu 2024 (SAO-YOLO)** | 85.7 % | 100 % | — | 12 TP / 2 FP |
| Dahi et al. 2017 (edge-based) | — | — | — | 10/11 videos clean; fails video 11 |
| **HLDNet (Kim 2022)** | — | — | — | 10/11 videos; fails video 11 (crowd) |
| Soontornnapar 2025 (Grounding DINO) | 83.33 % | 90.91 % | 86.96 % | fails videos 5 & 6 (night) |
| Lin et al. 2015 (origin) | 75.00 % | 81.82 % | 78.26 % | as re-reported by Soontornnapar |
| Altunay 2018 (Faster R-CNN) | 83.33 % | 100 % | — | indoor videos only (1,7,8,9,10) |
| Ngo & Mutaher 2025 (ViBe+CLIP) | — | — | — | CODR 100 % on 6/11, **0 %** on videos 7, 8, 11 |
| Dwivedi 2020 (contour) | 47.62 % | 90.91 % | 62.50 % | as re-reported |

---

### 1.2 Newer / larger public datasets

#### **IITP20** ✅ — the most substantial modern abandonment dataset

| Property | Value |
|---|---|
| **Origin** | **F. Amin, A. Mondal, J. Mathew**, *"A Large Dataset with a New Framework for Abandoned Object Detection in Complex Scenarios"*, **IEEE MultiMedia 28(3):75–87, 2021** |
| **Size** | **58 surveillance videos** |
| **Difficulty split** | **Easy (19)** — few people/objects, full view, but includes **objects on chairs/high platforms** and **radial distortion**. **Medium (20)** — adds groups of people and **person re-entry**. **Hard (19)** — hidden/partially-visible objects, **object exchanges**, **multiple owners for one object**, **one owner with multiple objects**. |
| **Annotations** | ⚠️ **No frame-level ground truth.** Videos are pre-sorted into `abandoned/` and `not abandoned/` folders by difficulty class only. The exact moment and nature of abandonment are not marked. |
| **⚠️ Label-quality problem (important)** | Vrsalovic et al. (PICom 2025) audited it and found **systematic semantic errors**: (a) luggage placed in **designated storage areas / lockers** is labelled "abandoned"; (b) **group-separation scenarios** where one member leaves but others remain with the bag are labelled "abandoned". They **re-derived the ground truth** and ran on the corrected version. |
| **Baseline** | Amin et al.'s own customised **RetinaNet** + object-association/owner-ID algorithm using **homography** for real-world distance. Reported ~**70 % AP** (IoU threshold not stated). |
| **Homography limitation** | Depends on a predefined ground plane → fails for objects on **elevated surfaces (chairs, platforms)**, which IITP20 deliberately contains. This is exactly what motivated Vrsalovic's depth-based radius. |

#### **CCTV-KD (Korzo + Düsseldorf)** ✅ — the most useful *annotated* in-domain set available

| Property | Value |
|---|---|
| **Origin** | Vrsalovic, Lerga & Ivasic-Kos, *Sensors* 25(9):2872, 2025 |
| **URLs** | **https://github.com/TheRomanFour/AbandonedLuggageDetection** · **https://app.roboflow.com/cars-0jbgu/luggage-person-detection-airport/** |
| **Size** | **474 annotated images, 9 174 object annotations (6 414 person / 2 760 luggage)**, avg 1 692 × 614 → resized 640 × 640 |
| **Sources** | **Düsseldorf Airport** (indoor, 240 imgs) + **Korzo pedestrian zone, Rijeka** (outdoor, 60 imgs), sampled at 1 fps from months of 24/7 footage across times of day |
| **Classes** | `person`, `luggage` |
| **Object sizes** | **6 976 small · 2 198 medium · 0 large** (COCO criteria) — the defining property of bird's-eye CCTV |
| **Annotation policy** | Only objects **> 10 px** and not heavily occluded were labelled |
| **Why it matters** | It is **annotated** (unlike ABODA), **in-domain** (unlike COCO), from **real public spaces** (unlike lab-staged sets), and **obtainable** (unlike PETS/AVSS). If you need a starting point for a detector, this is it. |
| **Caveat** | It contains **no abandonment events** — it is a *detection* dataset for person/luggage, not an *event* benchmark. Pair it with ABODA or IITP20 for the abandonment logic. |

#### **CDnet 2014 (changedetection.net)** ✅
- CVPR 2014 Change Detection Challenge dataset. Not abandonment-specific, but the **`intermittentObjectMotion`** category contains abandoned/static-object cases.
- Categories used in this corpus: *sofa* (2 people leave 3 objects), *abandonedBox* (outdoor, hard illumination), *streetLight* (cars stopping).
- Resolution used: 320 × 240. **Used by:** Dahi 2017 (bounding-box-level comparison vs. Wang et al. flux-tensor).

#### **VisDrone2019** ✅
- AISKYEY team, Tianjin University. **6 471 train / 548 val / 1 610 test**, ~2.6 M labels, 10 classes.
- **Not an abandonment dataset** — used purely as a **small-object-detection surrogate** because ABODA/PETS are too small to train a detector. **Used by:** Zhou & Xu 2024 (SAO-YOLO ablations + random-grouping error analysis).

#### **PETA (PEdesTrian Attribute)** ✅
- 19 000 images, **8 705 unique individuals**, 61 binary + 4 multi-class attributes.
- Contains **11 "carrying" labels**: BabyBuggy, Backpack, Other, ShoppingTro, Umbrella, Folder, LuggageCase, MessengerBag, Nothing, PlasticBags, Suitcase.
- Subsets used: **MIT, GRID, i-LID, CAVIAR4REID**.
- **Used by:** Soontornnapar 2025 to *verify a vision-language detector before deploying it*. Result is instructive — see §3.5.

#### **TCD (Trinity College Dublin)** ✅
- 2 videos of objects being abandoned and later collected (from Dawson-Howe's OpenCV textbook). Small and easy — every method in Smeureanu & Ionescu scores > 98 % on it.

#### **MS-COCO** ✅
- The universal pretraining substrate. Relevant classes: **`person`, `backpack`, `handbag`, `suitcase`** (Dogariu groups the last three into a single `baggage` class).
- ⚠️ **Domain-shift warning, quantified:** Vrsalovic et al. show COCO-pretrained detectors **collapse** on bird's-eye CCTV: **YOLOv8-m mAP@0.5 = 3.34 %**, YOLOv11-s/m/l = **3.2 % / 4.2 % / 1.1 %**, recall as low as **6.5 %**. After fine-tuning on ~470 in-domain images: **86 %**. This is the single most actionable number in the whole corpus.

#### **Anomaly-detection datasets (peripheral)** ✅
- **UMN** (Mehran 2009): 11 clips → one 4 min 17 s video, 7 739 frames, 640×480 @ 30 fps, 1 indoor + 2 outdoor scenes, temporal GT only.
- **CUHK Avenue** (Lu 2013): 37 videos (16 train / 21 test), 30 652 frames, 640×360 @ 25 fps, 47 abnormal events, temporal **and spatial** annotations.
- Used by Chaitra & Basthikodi 2023 — ⚠️ note these are **crowd-anomaly** datasets, not abandonment datasets (see the caveat in §3.5).

#### **CUHK03** ✅ — person re-identification, used by Dogariu et al. for the owner re-ID module (top-1 = 70.8 %).
#### **INRIA Person** ✅ / **MIT Pedestrian** ✅ — person-detector training (Altunay 2018; Lv et al. 2006).
#### **KISA** (Korea Internet & Security Agency) — Korean CCTV corpus with dynamic outdoor backgrounds (waves, moving vehicles, low light). Used by HLDNet as a *failure-mode stress test for dual-background methods*.

### 1.3 Custom datasets described in the corpus

Most are 🔒 not downloadable, but their **construction recipes are directly reusable**. **Note the exception at the top: the Vrsalovic CCTV sets ARE publicly released** (see the box below the table) — they are among the very few in-domain bird's-eye CCTV luggage datasets you can actually obtain.

| Dataset | Paper | Composition |
|---|---|---|
| **CCTV-Korzo** ✅ | Vrsalovic Sensors 2025 | 60 images (→100 after aug.), avg 942×526→640×640, outdoor pedestrian zone, Rijeka, Croatia. **1 423 annotations: 1 299 person / 124 luggage (8 %)** |
| **CCTV-Düsseldorf** ✅ | Vrsalovic Sensors 2025 | 240 images (→350), median 1 913×954→640×640, **indoor Düsseldorf Airport**, varied times/lighting. **7 751 annotations: 5 115 person / 2 636 luggage** |
| **CCTV-KD** (combined) ✅ | Vrsalovic Sensors 2025 | **474 images, 9 174 annotations (6 414 person / 2 760 luggage)**, avg 1 692×614. Object-size split: **6 976 small / 2 198 medium / 0 large** (COCO criteria) — i.e. **zero large objects**, the defining property of bird's-eye CCTV |
| **CCTV-KD-E** (extended) ✅ | Vrsalovic Sensors 2025 | 570 images, 11 104 annotations (7 890 / 3 214). *Adding these gave almost no gain (+1.3 mAP); cutting 35 % cost ~7 %.* |
| **Airport conveyor set** | Chen et al. IET 2024 | **> 4 600 original images**, 5 luggage classes: hard case (0), cardboard box (1), box w/ packing (2), soft bag (3), soft bag w/ packing (4) |
| **Highway abandoned objects** | Song et al. SIVP 2025 | **2 813 images** from Jiangsu highway surveillance (1280×960) + web imagery, 80/20 split |
| **Bag dataset** | Altunay et al. 2018 | **3 000 bag images** deliberately shot *in everyday scenes* (not on plain backgrounds) + a custom 5-video test set |
| **No Bag Left Behind** ✅ | Melebari et al. 2025 | 4 real-world scenarios; **code + data public: `github.com/ahmadmelebari/No-Bag-Left-Behind`** |
| **Stereo indoor set** | Beleznai et al. 2013 | **6 indoor sequences** (DOOR, MEETING, COFFEE, TABLE, CORRIDOR, LAB, TWO DOORS), 1 995–12 615 frames each, manually annotated. *"To our best knowledge there is no publicly available dataset for depth-based left object detection."* |
| **Tram RGB-D set** | Ajami & Lang | 8 OpenNI (.ONI) videos, synchronised RGB+depth, 2 camera positions inside a working tram, 4 scenarios |
| **CCTV building set** | Dogariu et al. 2020 | 1 hour, **downsampled to 1 fps → only 120 images** (motion-triggered cameras), basement/ground floor/exterior |
| **11-condition set** | Santad et al. 2018 | 11 videos varying **light source** (outdoor sun / fluorescent / window / blue), **camera height** (ceiling vs. eye level), **walk direction** (0°/45°/90° to camera plane) |

> ### ✅ The CCTV-Korzo / CCTV-Düsseldorf / CCTV-KD datasets are publicly available
> Vrsalovic, Lerga & Ivasic-Kos (Sensors 2025) release them in their Data Availability Statement:
> - **https://github.com/TheRomanFour/AbandonedLuggageDetection**
> - **https://app.roboflow.com/cars-0jbgu/luggage-person-detection-airport/**
>
> This matters a lot: they are **annotated** (unlike ABODA), **in-domain for bird's-eye CCTV** (unlike COCO), and come from a **real airport plus a real pedestrian zone** (unlike lab-staged sets). The Roboflow link in particular gives you the person/luggage annotations directly.
>
> **Ethics note worth reusing:** the authors obtained an ethical-review waiver on the grounds that the footage *"does not include any personal information but only object labels at the level of bounding boxes."* Funding: UNIRI project SAR-DAG (uniri-iskusni-drustv-23-278).

---
## 2. The problem definition everyone inherited

Almost every paper in this corpus — including 2026 ones — still uses the **PETS 2006 rule triple** verbatim or with small edits. Worth stating precisely, because your own definition choice determines what you can compare against.

**What counts as luggage:** anything hand-carryable — trunks, bags, rucksacks, backpacks, parcels, suitcases.

**Three rules (Thirde, Li & Ferryman 2006):**

1. **Contextual rule** — luggage is *owned and attended* by the person who enters the scene with it, until it is no longer in **physical contact** with them.
2. **Spatial rule** — after contact is broken, it is *attended* only while the owner is within **a = 2 m**; it becomes *unattended* once the owner passes **b = 3 m**. The band between *a* and *b* is a deliberate **warning zone** that absorbs calibration/detection error. All distances measured **between object centroids on the ground plane (z = 0)**.
3. **Spatio-temporal (abandonment) rule** — unattended for **t = 30 consecutive seconds** with no re-attendance by the owner *and* no contact by a second party (contact by a second party instead raises a **theft/tampering** event) → **alarm**.

**Variants that appear in the corpus — be aware these break comparability:**

| Variant | Who | Rationale |
|---|---|---|
| Alarm only when the **owner leaves the scene entirely** | Li et al. (I2R, PETS 2006); Liao/Chang | *"reduces false alarms significantly in busy public sites compared to the distance-based definition"* |
| **T_L = 60 s**, owner-left-scene | AVSS 2007 convention | i-LIDS house rule |
| **D > 2 × owner's bounding-box width**, held 5 s | Kim et al., HLDNet | Scale-free, avoids calibration entirely |
| **Bounding boxes simply must not intersect** | Dogariu et al. | Crude but cheap; no calibration |
| **Dynamic, depth-scaled radius** | Vrsalovic PICom 2025 | Fixes perspective: same pixel radius ≠ same metres |
| **10 s dwell, 40 px ownership radius** | Vrsalovic Sensors 2025 | Explicitly tunable per deployment |
| **τ = 75 stationary frames @ 15 fps ≈ 5 s** | Shah et al. 2026 | Much shorter than 30 s — inflates comparability problems |
| **CDist > 170 px AND ≥ 5 separation frames** | Soontornnapar 2025 | Purely 2D pixel thresholds |
| **Group ownership is transitive** | Vrsalovic (both) | If *any* group member stays, the bag is attended — explicitly rejects IITP20's labelling |

> **Design takeaway for your own work:** the *a/b/t* triple plus a "ground-plane centroid distance" measurement is the only definition with published ground-truth trigger times to validate against. Everything else is self-scored.

---

## 3. Per-paper deep dives

### 3.1 Era 1 — The PETS 2006 workshop (2006): geometry, tracking, and the birth of the benchmark

All eight papers below are inside `Computer_Vision_and_Pattern_RecognitionCVPR06_In_.pdf` (pages 47–106 of the printed proceedings).

---

#### **[P1] Overview of the PETS 2006 Challenge**
**Thirde, Li & Ferryman** — Computational Vision Group, University of Reading, UK · PETS 2006, pp. 47–50

- **Type:** Dataset/challenge definition paper (not a method).
- **Contribution:** Defines the left-luggage task, the three rules, the *a/b/t* parameters, the five luggage types, and the multi-sensor recording protocol at Victoria Station.
- **Dataset produced:** see §1.1 — 7 scenarios × 4 cameras, PAL 768×576 @ 25 fps, Tsai calibration + XML ground truth.
- **Why it matters to you:** this is the **origin of the operational definition** that 20 years of literature still uses. It's also the paper to cite for the *a=2 m / b=3 m / t=30 s* numbers.
- **Funding/legal:** EU ISCAPS (SEC4-PR-013800); ICO-approved for academic release.

---

#### **[P2] Left-Luggage Detection using Homographies and Simple Heuristics**
**Auvinet, Grossmann, Rougier, Dahmane & Meunier** — University of Montreal, Canada · PETS 2006, pp. 51–58

**Datasets:** PETS 2006, all 7 sequences, **all 4 cameras fused**.

**Methodology (3 stages, deliberately minimal-parameter):**
1. **Motion detection** — per-camera background subtraction. Background = **per-pixel median of 10 frames taken 1 s apart**. Threshold set so **≤ 1 % of background pixels misclassify** → a fixed **15 grey-level** threshold works for all cameras/sequences. Shadow removal via darkening-level + weak chromatic distortion. Then **1 erosion + 5 dilations (3×3)**.
2. **Fusion in the ground orthoimage** — instead of Tsai calibration (which they found gave *worse* orthoimages due to bad radial-distortion correction), they estimate a **direct 3×3 homography per camera** by least-squares from the supplied floor-point correspondences. Silhouettes are warped to the ground plane and **summed**. Blob threshold = **≥ 3 of 4 overlapping silhouettes**. The 5 dilations exist specifically to fix a **5–10 px foot-overlap error** caused by uncorrected lens distortion.
3. **Heuristic event recognition** on blob centroids only:
   - **Tracking:** blobs modelled as circles of radius ρ = √(N_pix/π); two blobs "touch" if ‖c₁−c₂‖ ≤ ρ₁+ρ₂ → same spatio-temporal entity. On collision, the entity with the longer history keeps the label.
   - **Spatio-temporal forks:** two non-touching blobs belonging to the same entity = an object separating from a person. *This is the core idea.*
   - **Immobile object:** a ground position with a blob within **30 cm every frame for > 3 s**; position = running mean of nearest blobs.
   - **Alarm:** fork + immobile branch + other branch > **b = 300 cm** for **30 s**.

**Results (PETS 2006, with shadow removal):** **6/7 true positives, 5 false positives total** (all 5 in S5, the ski-gear sequence). Spatial error **10.3–70.9 cm**; temporal error **+0.0 to +2.3 s**. Without shadow removal: 16 FPs, worse spatial error, +12.2 s on S5. **S3 correctly produced 0 detections** (negative control passed).

**Speed:** ~0.4 s/frame for pixel processing (C++/OpenCV, Centrino 2.26 GHz on 1200×400 orthoimages) + 0.02 s/frame tracking (Octave, Celeron M 1.4 GHz). **Not real-time.**

**Stated limitations:** circle-based blob association fails if motion is large relative to blob size; **no re-identification after merge/split** — if the owner meets another person and then one leaves, the algorithm cannot tell which. They propose colour histograms as the fix.

---

#### **[P3] Automatic Left Luggage Detection and Tracking Using Multi-Camera UKF**
**Martínez-del-Rincón, Herrero-Jaraba, Gómez & Orrite-Uruñuela** — Computer Vision Lab, University of Zaragoza, Spain · PETS 2006, pp. 59–66

**Datasets:** PETS 2006 — reports on **S1, S2, S3, S7**. **Cameras 1, 3, 4** (camera 2 rejected for poor resolution).

**Methodology:**
1. **Static object detection via double background subtraction:**
   - **Long-term background** = "clean" scene, initialised by temporal median, updated by temporal median over a set of short-term backgrounds.
   - **Short-term background** = last background patched with pieces of the current frame selected by a **static-object binary mask** = (current − last background) ∧ ¬(current − previous frame). Crucially the intersection is taken **at blob level, not pixel level** — if blobs from both subtractions touch, the whole blob enters the mask.
   - The two backgrounds' difference is **accumulated**; when the accumulator crosses a fixed-time value → static object.
2. **Luggage constraints:** area > A_min (=150, kills noise); size ≤ ½ a person's expected pixel size at that image point; **height/width ratio ≈ 1** (±5 %).
3. **Dynamic object detection:** current − long-term background only (short-term deliberately excluded so a person standing still doesn't vanish). Person = dynamic blob whose height matches a **height estimator** (±25 %); fragmented blobs are grouped inside the expected person bounding box.
4. **Scene calibration:** one homography per camera onto a station floor plan. **Height estimator** built from three vanishing points (2 horizontal + 1 vertical) using Criminisi's single-view metrology, with a 180 cm reference height — lets the system predict a person's pixel height anywhere in the image.
5. **Multi-Camera Unscented Kalman Filter (MCUKF)** — the paper's main contribution. A modified UKF where the **measurement-prediction stage is extended to S cameras simultaneously**: state vector n′ = r·S + e, with 2n′+1 sigma points. Per-camera Kalman gains are fused with weights **β⁽ˢ⁾** that combine (a) distance to prediction and (b) per-measurement covariance — interpretable as a per-sensor prior. Constant-acceleration motion model, state = [x, vx, y, vy]. Matching = nearest measurement per camera, rejected if χ² > 5.99 or distance > 2 m.
   - **Two independent trackers:** one fed by static-object blobs (luggage), one by dynamic blobs (owner). **They do not track everyone** — only the person nearest a detected static object.

**Results (PETS 2006, params a=2 m, b=3 m, t=30 s):**

| Seq | Distance error | Time error |
|---|---|---|
| S1-T1-C | 0.122 m | 0.06 s (1.5 fr) |
| S2-T3-C | 0.148 m | 0.06 s (1.5 fr) |
| S3-T7-A | 0.203 m | 0.18 s (4.5 fr) |
| S7-T6-B | 0.184 m | 0.10 s (2.5 fr) |
| **Mean** | **0.164 m** | — |

**Thresholds published:** T1=10, T2=T4=T5=30, T3=230, T_min=100, A_min=150, r_H/W=±5 %, r_std_H=±25 %.

**Design philosophy worth stealing:** they explicitly argue *against* tracking everyone — "would demand unnecessary computational resources." Detect the abandonment first, then track backwards to the nearest person.

---

#### **[P4] Multi-View Detection and Tracking of Travelers and Luggage in Mass Transit Environments**
**Krahnstoever, Tu, Sebastian, Perera & Collins** — GE Global Research, Niskayuna NY · PETS 2006, pp. 67–74

**Datasets:** PETS 2006 — **S1, S2, S3, S4, S6** (S5 & S7 excluded: focus on *small* luggage). **2 of 4 views** used; one view rejected for shooting through a glass wall, another for horizon-in-mid-frame geometry. Ran on MPEG-4 transcodes of the JPEG sequences.

**Methodology — the key architectural idea is scalability by separation:**
- **Detection is local (per camera view); tracking is central (in the calibrated ground plane).** Detections are timestamped on a synchronous clock, buffered, and time-reordered because the central tracker sits on a separate node.
- **Geometry-driven detection:** targets modelled as **vertical ellipsoids** on the ground plane with class-specific height/width. Bounding boxes of these ellipsoids approximate silhouettes; a person box is split into **3 parts** (top/middle/bottom third). The foreground likelihood is written as a log-likelihood over per-part histograms of likelihood ratios, then **approximated so it evaluates via integral images** — reducing each hypothesis to *~4 memory lookups + 4 additions*. Search is **greedy** (add the target that most increases likelihood, then spatially prune) rather than MCMC, explicitly for real-time.
- **Target classes:** adult (0.5 m ⌀, 1.8 m h, min 20×20 px), child (0.4 m, 1.3 m, 20×20 px), luggage (0.5 m, 0.5 m, 25×25 px). ~161 000 candidate locations evaluated per frame in camera 4, ~82 000 in camera 3.
- **Refinement:** head-location search to refine person ground positions (filters out the classic false alarm of a tall person's head projecting far from their feet).
- **Central tracker:** generalized nearest-neighbour — track prediction → **Munkres/Hungarian assignment** under **Mahalanobis distance** → **extended Kalman filter** update with a **constant-velocity-turn** model → track maintenance (delete on uncertainty/out-of-view/no-association) → track formation with size gating and **'ghost' patching**.
- **Event logic:** every new luggage track is bound to the **nearest person within r_o = 1 m** — *"the system does not allow spontaneous discovery of abandoned luggage"* (deliberate FP suppression). Luggage becomes "stationary" when its location covariance over **τ_s = 3 s** falls below **r_s = 0.2 m**. Warnings at a = 2.0 m and b = 3.0 m; alarm after **τ_u = 30 s**. Pickup = track disappears/moves while a person is within **r_p = 1.0 m**; pickup by a non-owner → **theft alarm**.
- **Calibration:** they **re-calibrated** rather than use the supplied Tsai parameters, using Bayesian autocalibration + interactive ground-plane landmarks; lens distortion not compensated.

**Results:**

| Seq | Owner leaves | Warning | Alarm | Comment |
|---|---|---|---|---|
| S1 | 2059 | 2088 (1.1 s) | 2854 (31.8 s) | backpack abandoned ✓ |
| S2 | 1512 | 1542 (1.2 s) | 2308 (31.8 s) | ✓ despite a worker moving 3 garbage bins behind the glass wall |
| S3 | n.a. | n.a. | n.a. | ✓ correctly no alarm |
| S4 | 1802 | 1845 (1.7 s) | 2611 (32.4 s) | ✓ track-linking held the luggage ID while two men stood over it |
| S6 | 1637 | 1689 (2.1 s) | 2455 (32.7 s) | ✓ **plus 1 false alarm at 1336** |

**All abandonment events detected, 1 FA.** Speed: **2 × 720×576 streams at 15 fps on a single-core 3 GHz Pentium 4**.

**The false alarm is the most instructive result in the whole corpus.** A man sat on a bench reading a magazine for the entire sequence. Because the system initialised on a non-empty scene, he was never detected as a person; his **arm movements** generated spurious object detections, which got associated with a *different* person standing behind him — and when that person left, the alarm fired.

**Their conclusion (still true 20 years later):** *"much larger datasets are needed… PETS 2006 is only moderately complex compared to, for example, a crowded gate in an airport"*, and *"the problem of long-duration stationary (e.g. sitting) people is one of the major challenges."* They argue for **strong non-adaptive but lighting-invariant background models** — the system needs a persistent notion of the empty scene.

---

#### **[P5] Detecting Abandoned Luggage Items in a Public Space**
**Smith, Quelhas & Gatica-Perez** — IDIAP / EPFL, Switzerland · PETS 2006, pp. 75–82 *(also present as the standalone `SmithQuelhasGatica-cvpr-pets06.pdf`)*

**Datasets:** PETS 2006, all 7 sequences, **camera 3 only**, images **downsampled to half resolution (360 × 288)** for speed. Because the tracker is stochastic, **5 runs per sequence** and means are reported — the only paper in the corpus that reports run-to-run variance.

**Methodology — two-tier: generic tracking, then bag reasoning.**

*Tier 1 — Trans-dimensional (Reversible-Jump) MCMC tracking:*
- Mixed-state Dynamic Bayesian Network jointly modelling **the number of objects and their locations/sizes**. State per object Xᵢ,ₜ = (x, y, s_y, e) = position, height scale, eccentricity. Zero objects is a special state ∅.
- Dynamics: 2nd-order autoregressive per object + a **pairwise MRF interaction potential** that stops two trackers fitting the same object.
- Observation model on a single foreground-segmentation source (Stauffer–Grimson): a **multi-object likelihood** = response of a 2-D Gaussian centred at a learned point in **precision-recall space** (ν, ρ) per object, normalised by mₜ; plus a **zero-object likelihood** = exp(−λ·max(0, U−B)) over **weighted uncovered foreground pixels**, which pushes the model to place a tracker on every large enough foreground patch (this is what births new objects).
- Inference by **RJMCMC with three moves: birth, death, update**, Metropolis–Hastings acceptance with dimension matching. Solution = mean of the chain.
- **Critically: the tracker does not distinguish people from luggage.**

*Tier 2 — Left-luggage detection process:*
Built on three assumptions: left luggage (1) probably doesn't move, (2) probably looks smaller than people, (3) **must have an owner**.
- **Step 1 — find bags.** Object blobs = tracker box ∩ foreground. 5-frame sliding window → velocity. Bag likelihood **p(Bⁱ=1) ∝ Σₜ N(sₜⁱ; μ_s, σ_s)·exp(−λ vₜⁱ)** — small + slow + long-lived, **summed without normalising by lifetime**. Threshold T_b. Then reject candidates at image borders and candidates stacked on other candidates.
  - **Bag lifespan** via a **shape template** T ⁱ built from the longest low-velocity segment (binary foreground patches, background pixels set to −1), and a **bag-existence likelihood** = elementwise product of template and current binary image; threshold = 80 % of the max.
- **Step 2 — find owner.** Inspect the tracker present when the bag first appeared. If it moves away and dies while the bag stays → that's the owner. If it stays with the bag and dies → search for **new track births within radius r = 100 px**; first birth = owner. No nearby birth → assumption 3 violated → **discard the bag**.
- **Step 3 — alarm.** 2-D **homography (DLT)** from image to station floor, using the PETS floor pattern. Since blob centroids aren't on the floor, they estimate the **foot point** = (mean x, bottom-most y) and pass *that* through the DLT.
- Note: **can run online** using the 30 s alarm window as the search horizon.

**Parameters published:** μ_s=380, σ_s=10000, λ=10, T_v=1.5, T_b=5000, r=100, b=3, B=800. Foreground precision parameters were learned by annotating **41 bounding boxes from S1 and S3** and perturbing them — an honest admission of test-set training.

**Results (mean of 5 runs), evaluated with a speech-recognition-style error rate (deletions+insertions)/events:**

*Luggage detection (bags placed on the floor, even if never abandoned):*

| Seq | GT | Mean detected | Error | Spatial error |
|---|---|---|---|---|
| S1 | 1 | 1.0 | 0 % | 0.16 m |
| S2 | 1 | 1.2 | 20 % | 0.22 m |
| S3 | 1 | 0.0 | **100 %** | — |
| S4 | 1 | 1.0 | 0 % | 0.32 m |
| S5 | 1 | 1.0 | 0 % | 0.13 m |
| S6 | 1 | 1.0 | 0 % | 0.40 m |
| S7 | 1 | 1.0 | 0 % | 0.19 m |

*Alarm detection:* **correct in 6/7** — S1 (0.78 s error), S2 (1.08 s, +1 FP warning), S3 (correct 0 alarms), S5 (0.04 s), S6 (0.08 s), S7 (3.56 s). **S4 = 100 % error**: the second actor stayed next to the bag, and the model **repeatedly mistook him for the owner**, so no alarm ever fired. They attribute this to the tracker not modelling colour.

**Notable failure causes:** S2 FP came from **trash bins being moved**, disrupting the segmentation. S3's 100 % luggage-detection error is arguably a definitional artefact — the owner never left, so no "bag-like blob" was ever produced, yet the system correctly predicted 0 alarms.

---

#### **[P6] Left Luggage Detection using Bayesian Inference**
**Lv, Song, Wu, Singh & Nevatia** — Institute for Robotics and Intelligent Systems, USC · PETS 2006, pp. 83–90

**Datasets:** PETS 2006, all 7 sequences, **camera 3 only** ("best viewpoint; single-view results are satisfactory").

**Methodology — the cleanest separation of low-level tracking from high-level event reasoning in the corpus.**

*Tracking = fusion of two complementary trackers:*
1. **Kalman-filter blob tracker.** Background learned from the **first 500 frames**. Blob = bounding box + colour histogram. Association value = IoU-style overlap of predicted vs. observed boxes, thresholded. Handles 5 cases explicitly: 1-to-1 update; unmatched blob → new object; object unmatched for N frames → removed; object with multiple blobs → merge; **blob with multiple objects → segment the blob by each object's colour histogram**. **Luggage is identified purely by mobility** — "it seldom moves after the start of its track."
2. **Detection-based human tracker.** Nested cascade detectors of **boosted edgelet-feature weak classifiers** (Wu & Nevatia). Three full-body detectors: frontal/rear, left-profile, right-profile (= flipped left). **Training set: 1 700 positives frontal/rear, 1 120 left-profile, 7 000 negatives**, sourced from the Internet + the **MIT pedestrian set** — *fully independent of the test sequences*, i.e. generic. Data-association tracking with initialisation/termination confidences from colour+shape+position, falling back to a **colour mean-shift tracker** when detection fails.
3. **Fusion rule:** if a human-tracker trajectory largely overlaps a blob-tracker trajectory, **the human trajectory supersedes it** (shape-based ⇒ more reliable); the blob tracker contributes the non-human objects the detector misses. This resolves both failure modes: merged human blobs, and detector failure at large tilt angles.

*Event recognition = Bayesian inference over trajectories:*
- Trajectories are median-filtered, then mapped **2-D → 3-D** by assuming the lowest point is at z = 0 and using the camera model on the bottom-centre of the box.
- Events are **hypotheses**, cues are **evidence**; competing events are competing hypotheses. Posterior via Bayes with conditional independence of evidence. Distributions available: Histogram, Threshold, Uniform, Gaussian, Rayleigh, Maxwell — learned from data when available, else user-specified. Default P(H)=0.5 when unknown.
- **Drop-off luggage** — 3 pieces of evidence: (E1) luggage did not exist Δt ago; (E2) luggage exists now; (E3) person–luggage distance now < D_thres (e.g. 1.5 m). E1∧E2 = temporal constraint (the bag track only begins at drop-off); E3 = spatial constraint eliminating passers-by. If several people qualify, **the nearest is the owner**.
- **Abandon luggage** — 2 pieces: distance Δt ago < D_alarm (3 m); distance now > D_alarm.
- **Temporal rule** implemented as suppression: if a previous frame within 30 s already had high abandon probability, the current one is discarded.
- **Warning** = identical model with a 2 m threshold.

**Results:** **all warning and alarm events detected within 9 frames (0.36 s)** of ground truth. Correct number of involved persons in every sequence **except the last**, where three people walked as a tight group and one was severely occluded.

**The paper's most valuable finding — an ablation on the contextual rule:** they show a table of triggered events *without* the contextual rule (drop-off filtering) and it is full of false alarms. *"The contextual rule is crucial because detecting the drop-off event provides a filtering mechanism to disregard the irrelevant persons and luggage… high-level reasoning can eliminate the errors of low-level processing."* They also note the whole left-luggage capability was obtained **just by specifying event models** in an existing general event-recognition framework — done immediately, no retraining.

---

#### **[P7] Evaluation of an IVS System for Abandoned Object Detection on PETS 2006 Datasets**
**Li, Luo, Ma, Huang & Leman** — Institute for Infocomm Research (I²R), Singapore · PETS 2006, pp. 91–98

**Datasets:** PETS 2006, all 7 sequences, **camera 3**, subsampled to **1 frame in 3 (~8.33 fps)** and downscaled to **176 × 144** — the lowest resolution in the corpus. Justification: (a) real-time on their deployed hardware, (b) *less fragmentation* in segmentation at low resolution, which matters more than detail. On the other three cameras the left objects were indistinguishable from segmentation noise at that resolution.

**Methodology — four modules, and unusually, a system that had already been field-deployed.**
1. **Foreground segmentation:**
   - **PFR (Principal Feature Representation) adaptive background subtraction.** Per pixel, a table of the N_v most frequent feature vectors + their statistics; classification by a **Bayes rule**. Three feature types: **spectral** (RGB colour), **spatial** (Sobel gradient), **temporal** (colour co-occurrence (R,G,B)ₜ₋₁,(R,G,B)ₜ at 32 levels/channel). Static pixels use colour+gradient; **dynamic pixels use co-occurrence**.
   - **Context-controlled background maintenance** — the distinctive part. Two kinds of *contextual background regions*: **Type-1** = fixed facilities (ATM, counter, chair) described by **Orientation Histogram Representation + Principal Colour Representation**, fused log-likelihood (ω_s = 0.6); **Type-2** = large homogeneous regions (ground, walls) described by PCR only, evaluated over 5×5 windows. Three learning rates: **α = 0** (freeze) where a CBR is occluded, **α** normal, **2α** to *recover* a background model where a CBR is visible but a foreground region overlaps it. This is a principled answer to background poisoning.
2. **Moving-object tracking** by **Principal Colour Representation (PCR)**: object = size + N most significant (RGB, significance) pairs. Likelihood uses a min-based colour-match sum, plus a scale-normalised variant, taking the max (**scale-adaptive likelihood**). Frames linked as **directed acyclic graphs** (parent = previous-frame regions, child = current); multi-object assignment posed as **MAP**, decomposed coarse→fine: depth-order estimation, sequential assignment most-visible-first, **exclusion** (removing an assigned object's colour evidence from the region) after each iteration; then PCR-based **mean-shift** for fine location.
3. **Stationary-object tracking** by a **layer model / template**: once a small stationary object is detected, its image is frozen at a fixed position. Per-frame colour+gradient differences → fuzzy measures with 2σ̂ as the 0.5 point. Low difference = exposed (template slowly updated); high difference + no overlapping foreground = **removed**; high difference + overlapping foreground = **fully occluded**; moderate + partial overlap = partially occluded. **This is what lets a small bag survive complete occlusion.**
4. **Event detection** by **Finite State Machines** — chosen for efficiency/flexibility. The Unattended-Object FSM: INIT (new small object separated from a large moving object; ownership established) → Station (object becomes stationary, associated with owner) → **UO** (owner leaves scene) / Cancel (moves again or disappears).

**Alarm definition:** deliberately **owner-out-of-scene**, not distance-based — *"reduces the false alarms significantly in busy public sites."*

**Results:**
- **Left-luggage accuracy 5/7 = 71.4 %, zero false alarms** — **with the same parameter set used in real deployments, no scene-specific tuning.**
- Failures: **S2** (background changed as a trolley coach moved ~10 s before the actors left → cluttered foreground → tracking errors); **S5** (the ski carrier "looks like a side view of a standing person"; allowing that detection would produce many real-world FPs).
- **With two scene-specific additions** — removing foreground behind the fence via calibration, and classifying a human-shaped object motionless > 30 s as non-human — **7/7 = 100 %, still no FPs.**
- **Tracking evaluation:** 123 valid objects across the 7 camera-3 sequences, **9 ID changes → 7.32 % tracking error rate**.
- **Deployment context:** the system had run **around the clock for three months at several busy public sites**, at **176×144, 10 fps on a 2.8 GHz PC**. Demos: `perception.i2r.a-star.edu.sg/PETS2006/UnAttnObj_C3.htm`.

---

#### **[P8] Abandoned Object Detection in Crowded Places**
**Guler & Farrow** — intuVision Inc., Woburn MA (US Government funded) · PETS 2006, pp. 99–106

**Datasets:** PETS 2006 — **all 28 videos (4 cameras × 7 scenarios)**, the most complete use of the benchmark in the corpus. Camera calibration via **Tsai** using the supplied ground-plane info; calibration error **0.08 m for cameras 3 & 4**, **0.2 m for cameras 1 & 2**.

**Methodology — the "two detectors in parallel" pattern that later papers rediscover repeatedly.**

Their framing of the real problem: *"especially when there is harmful intent, these events do not happen in an obvious manner… The owners usually move around the object they are going to leave behind pretending checking schedules, or asking someone a question."*

1. **Moving-object tracker** — MoG background subtraction per RGB channel; connected components; shadow removal by separating brightness from chromaticity; frame-to-frame correspondence by motion+position+region, with an appearance model (size, aspect ratio, colour distribution). Positions taken as the **centre of mass projected onto the ground plane**. Tracks are analysed for **object splits**, and each split is qualified as a candidate **drop-off** by the sizes and motion of the spawned object and the parent–child ground distance.
2. **Stationary-object detector (the novelty)** — runs **in parallel on the same background model**, producing a **stationary-object confidence image S** ∈ [0,255]:
   - `s(x,y) += C(x,y)·(255/(t·FrameRate))` when i(x,y) doesn't fit b(x,y)(μ,σ)
   - `s(x,y) −= r·D(x,y)·(255/(t·FrameRate))` when it does
   - where **C** = consecutive non-fitting observations, **D** = consecutive fitting observations, **t** = drop-off wait time, **r** > 0 controls how fast confidence decays.
   - **The key property:** pixels on an abandoned object stay different from the background *regardless of which object they belong to* — so a **moving crowd passing in front does not destroy the detection**. Object removal turns confidence off in t/r seconds. Colour distributions can be built during confidence accumulation to distinguish the true stationary object from occluders.
3. **Fusion + multi-camera voting** — candidate drop-offs are validated against detected stationary objects (dual mechanism, reduces view dependence). Warning/alarm from owner distance per PETS rules. Then **cross-camera correlation**: warnings correlate if within **t seconds** and the object coordinates within **d metres**; two correlated cameras → a global warning, and the first subsequent alarm → global alarm.

**Parameters (Table 3):** drop-off event wait time **3 s (cameras 1,2,4) / 2 s (camera 3)** — the *only* parameter changed across views; drop-off idle distance **3 px**; stationary-object overlap **80 %**; PETS-defined 2 m / 3 m / 30 s.

**Results — single-camera (28 experiments, 21 with identical parameters):**
- **Cameras 3 & 4** (high, top-down): best; average alarm-time error **0.8 s**. Worst case = camera 4 / scenario 1, where the parent walked **toward** the camera, delaying the drop-off detection until after the 3 m crossing → **3.27 s late**.
- **Cameras 1 & 2** (low mounting): promising but harder — camera 1 avg **2.14 s** and **0.40 m** error, hurt by multiple occlusions, shadows and **reflections displacing the computed ground position**; 3 FPs total in camera 1 (1 in scenario 6, 2 in scenario 7).
- **Scenario 4** exposes the structural limit of split-based analysis: *the owner did not leave until another person stood next to the bag*, so no alarm — but the bag's location was still recovered to **0.21 m** from the stationary image.
- **Camera voting results:** scenario 1 → 0.16 s / 0.057 m; scenario 2 → 0.04 s / 0.252 m; scenario 5 → 0.20 s / 0.272 m; scenario 6 → 0.04 s / 0.139 m; scenario 7 → 1.04 s / 0.138 m. **Voting successfully eliminated the bad single-camera results** (camera 4/S1, camera 4/S6, camera 1/S7).

**Proposed future work that later papers adopted:** if the drop-off event is unobservable, run a **real-time historic search** back to the frame where the abandoned object first appeared, find the closest object then, and designate it the owner. (This is exactly Lin et al. 2015's "back-tracing verification" and Liao's "selective tracking".)

---
### 3.2 Era 2 — Classical CV maturity (2008–2017): localisation, edges, depth, hand-crafted classifiers

---

#### **[P9] A Localized Approach to Abandoned Luggage Detection with Foreground-Mask Sampling**
**Liao, Chang & Chen** — DSP/IC Design Lab, National Taiwan University · **AVSS 2008**, pp. 132–139

**Datasets:** **AVSS 2007** (Easy/Medium/Hard) + **PETS 2006** (all 7).

**Methodology — three stages; the core idea is "localise first, then track only what matters."**

*Stage 1 — Foreground-mask sampling.* Take **6 frames evenly spread over the past 30 s** (the temporal rule window). Background-subtract each with a **row-dependent weight** on the standard deviation: `F(i,j)` is foreground iff `|F(i,j) − B(i,j)| > w(i,j)·Std(i,j)`, where **w increases with image row** — this compensates for the resolution gradient caused by a tilted, downward-looking camera (objects near the top are smaller/lower-resolution). Binarise the 6 masks and take the **pixel-wise intersection** `S = M₁·M₂·…·M₆`. A white region in **S** has been foreground in all 6 samples over 30 s ⇒ either an abandoned object or a motionless human. Filter noise + connected components.
   - **Why this is elegant:** no appearance model, no prior learning ⇒ **luggage of any shape, size, orientation, viewing angle or colour** is localisable.

*Stage 2 — Selective tracking.* For each static region, decide human vs. luggage using two cues:
   - **Cr colour channel (YCbCr)** for skin — chosen because *the face is the most visible body part under a downward-tilted camera in a crowd*, and skin response in Cr is strong regardless of race. Background subtraction is done **inside the search region in RGB first**, then converted to YCbCr, so the face signal is stronger with clutter removed.
   - **Improved Generalized Hough Transform on the head–shoulder contour.** Template generation records (ψ, r, α) into a **180-bin reference table**; matching accumulates votes on a detection map. **Their improvement:** instead of accessing only bin (m+1), access **11 bins (Δm = 5) with Gaussian weights**, because ψ computed from a quantised edge point is only indicative of a small angular *range*. Shown superior to plain GHT and to normalised correlation (fewest false positives).
   - If the region is human → discard. If luggage → search the neighbourhood for the owner in the current frame. **If the owner is not there, go back in time Δt = 60 s** to when they were, and track forward from there. Only the owner is tracked — *selective* vs. *comprehensive* tracking. This explicitly avoids identity switches.
   - **Motion prediction:** r(t+1)=r(t)+Δr with Δr = α·Δr + β·(r(t)−r(t−1)), **α = 0.4, β = 0.6** — exponentially decaying history so the predictor follows speed changes.

*Stage 3 — MAP probabilistic event model.* Trajectory reliability score `P_TOTAL = λ_POS·P_POS + λ_SIZE·P_SIZE + λ_CH·P_CH` combining position, size (pixel area), and **Bhattacharyya distance between colour histograms** of prediction vs. detection. Event A = abandonment, O = the observed trajectory. Prior **P(A)=0.5**, **P(O|A)=0.95**, and **P(O)** modelled as the mean of P_TOTAL over all processed frames. Then `P(A|O) = P(O|A)·P(A)/P(O)`, and an alarm fires if it exceeds a user-adjustable **ρ**. The user-tunable threshold is the stated advantage — sensitivity becomes a policy dial, and the alarm carries a confidence.

**Results:**
- **AVSS 2007: 3/3 detected.** Easy — owner tracked continuously until leaving. **Medium and Hard — the owner is occluded by a large pillar for ~1.5 s (~40 frames at 25 fps); the tracker loses him** and the object is declared lost, so an alarm fires *anyway* (arguably right answer, wrong reason).
- **PETS 2006: alarms correctly issued in videos 1, 2, 4, 5, 6.** Video 3 correctly silent. **Video 7 — the owner wanders with abrupt speed/direction changes; the motion predictor cannot follow. The owner is lost 34 s after leaving, but the alarm fires at 30 s** (i.e. the 30 s rule saved it).

---

#### **[P10] Localized Detection of Abandoned Luggage** *(journal extension of P9)*
**Chang, Liao & Chen** — NTU · **EURASIP JASP 2010, Article ID 675784**

Same two core techniques (foreground-mask sampling + selective tracking), with added engineering detail and **quantitative timing**.

**Datasets:** AVSS 2007 + PETS 2006. Note the **alarm-definition mismatch** they flag: `T_L = 60 s, owner-left-scene` in AVSS 2007 vs `T_L = 30 s, owner-left-luggage` in PETS 2006.

**Background model:** average of hand-selected clean frames (or minimal-clutter frames when clean ones don't exist), with a per-pixel standard deviation, plus **two filters after the intersection** to remove non-luggage static regions. ⚠️ **They deliberately use no dynamic background update**, justified because *"the tested video sequences contain minimal ambient lighting change"* — a real limitation for outdoor deployment, though they note the module is swappable for any other background subtractor.

**The row-weight formula, now explicit:** `w(i,j) = (c/h)·i·W`, where *h* = image height, *c* = number of quantisation steps, *W* = the weight on top-most pixels. Any monotonically increasing function of row *i* can replace it; quantisation keeps the upper/lower weight ratio reasonable.

**⚠️ Parameter change vs. the 2008 conference version:** the journal sets the back-tracking window **Δt = 30 s** (the AVSS 2008 paper used **60 s**), justified by the assumption that when an isolated luggage item is first detected, its owner must have been close to it until shortly before detection.

**Two refinements not in the conference version:** (a) **several differently-sized head-shoulder templates are applied simultaneously**, since people at different image locations have different silhouette sizes; (b) the colour-histogram distance is specifically the **χ² distance**, `D² = Σᵢ₌₁²⁵⁶ (c_P(i) − c_D(i))² / (c_P(i) + c_D(i))`, chosen empirically. They note the three probability scores *"serve more as comparative than absolute values"*, so the standard-deviation choices have insignificant effect on ranking.

**A robustness claim worth noting:** because skin colour *and* head-shoulder contour must both agree, *"even [if] the color of abandoned object is close to skin color, the object will not be recognized as a human since it has no head-shoulder contour."*

**Alarm timing results (seconds):**

| Sequence | Owner break | Left scene | G.T. | Alarm | **Diff** |
|---|---|---|---|---|---|
| AVSS2007 Easy | 114.80 | 119.76 | 180.00 | 179.76 | **−0.24** |
| AVSS2007 Medium | 100.88 | 102.64 | 162.00 | 162.64 | **+0.64** |
| AVSS2007 Hard | 101.08 | 102.28 | 162.00 | 162.28 | **+0.28** |
| PETS2006 Seq.1 | 85.88 | 90.52 | 113.72 | 120.52 | **+6.80** |
| PETS2006 Seq.2 | 61.92 | 65.04 | 91.84 | 95.04 | **+3.20** |
| PETS2006 Seq.4 | 72.88 | 76.36 | 104.08 | 106.36 | **+2.28** |
| PETS2006 Seq.5 | 80.28 | 83.04 | 110.56 | 113.04 | **+2.48** |
| PETS2006 Seq.6 | 68.44 | 73.96 | 96.88 | 103.96 | **+6.08** |
| PETS2006 Seq.7 | 60.68 | — | 93.96 | 91.60 | **−2.36** |

*("Owner break" = the moment the owner separates from the luggage; "Left scene" = the moment they exit camera view — S7's owner never cleanly leaves, hence the dash.)*

**Max |error| = 6.80 s; mean |error| = 2.71 s.**

**Head-to-head comparison tables:**

| AVSS 2007 | Tested events | True detections | False alarms |
|---|---|---|---|
| Porikli et al. (dual foregrounds) | 3 | 3 | **2** |
| Bhargava et al. | 8 | 8 | **4** |
| **This method** | 3 | 3 | **0** |

| PETS 2006 | Tested events | True detections | False alarms |
|---|---|---|---|
| Porikli et al. | 1 | 1 | 0 |
| Bhargava et al. | 6 | 6 | 0 |
| **This method** | 6 | 6 | 0 |

**Speed: 17.37 fps average on D1 (720×576), 2.66 GHz Intel E6750 + 4 GB.** *Real-time, in 2010, on a CPU.*

**Their explanation of the FP advantage:** pure background-subtraction methods without a human detector raise false alarms when a person stops moving briefly; the head-shoulder + skin detector rejects those.

**Honest limitations stated:** multiple simultaneous abandoned objects require multiple selective trackers → frame rate drops; **high crowd density remains unsolved** — *"even humans cannot notice abandonment reliably. In this case, foreground-mask sampling may fail and the system needs an object-recognition-based solution."*

---

#### **[P11] Automatic Classification of Abandoned Objects for Surveillance of Public Premises**
**Otoom, Gunes & Piccardi** — University of Technology Sydney · **CISP 2008**, pp. 542–547

**A different problem: not detection but *classification* of the already-detected object.** Their argument: in airports/stations abandoned objects are mainly **luggage or trolleys**, and *no prior work had attempted to recognise trolleys*.

**Datasets:**
- **PETS 2007**: 125 images of empty/loaded trolleys, bags, persons, groups — deliberately **one image per distinct physical object** (drawn across S1–S8) to avoid the same object appearing in both train and test folds. They explicitly note PETS 2007 has only **2 abandonment scenarios (S7, S8)**, insufficient for a classification study.
- **Mixed set**: 184 images = **124 uncluttered** (downloaded from the web, 31 per class) + **60 cluttered** (clipped from real airport videos supplied by an industrial partner).
- Combined: **309 images**.
- They work from **manually cropped** objects to isolate classification error from detection error.

**Four classes:** trolley(s), bag(s), single person(s), group(s) of people.

**Features (all shape-based, since abandoned objects don't move) — extracted with OpenCV:**
- **Corners:** count; percentages/ratios across image quadrants; horizontal & vertical std-dev of corner distribution.
- **Lines** (edge detector + Hough): count of strong/intermediate/weak; counts of horizontal/vertical/diagonal and their ratios.
- **Circles** (Hough): count; ratios across image parts; std-dev of positions and radii.
- **Compactness** = perimeter²/area, normalised by image size.
- **Height/width ratio** (raw height and width rejected as scale-dependent).

**Class rationales:** trolleys = many closely packed strong straight lines + many corners + circles in the *lower* half (wheels); person = intermediate corner count at head/hands/legs, few vertical lines depending on posture, **one circle in the upper half (head)**; group = several head-circles close together, high circle std-dev; bag = corners at handles/zippers/wheels, few boundary lines, some circles at handle/wheels.

**Classifiers (WEKA):** **BayesNet**, **C4.5** decision tree, **SMO** (SVM). 10-fold CV + holdout.

**Results — Experiment 1 (invariance):**

| Classifier | PETS 2007 | Mixed |
|---|---|---|
| BayesNet | 70.4 % | 67.9 % |
| C4.5 | 72.0 % | 66.9 % |
| SMO | 68.8 % | **73.1 %** |
| **Average** | **70.4 %** | **69.3 %** |

The near-identical averages are their main claim: the feature set is **invariant to dataset and to classifier**, so no re-tailoring is needed per deployment.

**Experiment 2 — occlusion handling:**
- **Temporal occlusion** (moving occluder). 111 consecutive frames of the abandoned object in **PETS 2007 S8**, sampled every 2nd frame → 56 images, of which **32 are partially or fully occluded**. **Multi-frame integration** — sum binary per-frame decisions over T frames, take argmax: `x* = argmax Σ d(x|fᵢ)`. **Result: 94 % (53/56 frames correctly 'bag').**
- **Spatial occlusion** (static occluder). A trolley fragment from PETS 2007 was **manually overlaid** in front of the abandoned bag across 30 frames (occluded for 9 consecutive). **Result: 90 % (27/30)**. Under extensive occlusion the bag was misclassified as 'trolley' or 'group of people'. **They give a concrete threshold: *"when the object is occluded for more than 2/3, as the features are not accurately extracted the classifier outputs incorrect results."*** Per-frame outcomes were: no occlusion → Bag, partial → Bag, **full → Trolley** — and multi-frame integration recovers the correct 'bag' verdict.

- **An honest confound they raise themselves** on the temporal-occlusion result: the 94 % is partly explained by the fact that *"the occluding object was mostly a person carrying another bag, thus the object was still containing typical features of class 'bag'."* Per-frame outcomes there were: no occlusion → Bag, **partial → Person**, full → Bag.

- **Generalisation caveat, in their words:** *"the results obtained in these experiments relate to these two cases and cannot be easily generalized to other conditions or types of abandoned objects without extensive experimentation. The recognition accuracy might vary significantly if the abandoned and the occluding objects are of different nature or type."*

**Relevance to you:** this is the only paper in the corpus that treats **"what kind of object is it"** as a first-class problem with a documented feature set and public-benchmark evaluation. 70 % four-class accuracy is a realistic ceiling for hand-crafted shape features at surveillance resolution.

---

#### **[P12] Reliable Left Luggage Detection Using Stereo Depth and Intensity Cues**
**Beleznai, Gemeiner & Zinner** — AIT Austrian Institute of Technology, Vienna · **ICCV Workshops 2013**, pp. 59–65

**The strongest sensor-side contribution in the corpus.**

**Sensor:** in-house **trinocular** stereo rig — three parallel monochrome board-level industrial cameras, USB 2, **40 cm baseline** between the outer two, **1280×1024 resampled to 1150×920, 8-bit**, calibrated offline. Depth by a **pyramidal Census-transform stereo matcher optimised for embedded real-time**, computed **for all three available baselines** to improve quality across spatial ranges. **~10 fps** for stereo alone on a modern PC.

**Datasets:** 🔒 **6 self-recorded indoor sequences** — they state explicitly *"To our best knowledge there is no publicly available dataset for depth-based left object detection."* Each sequence was designed to embed a known failure mode:

| Sequence | Frames | Static regions | True left objects | Complexity injected |
|---|---|---|---|---|
| DOOR | 1 995 | 1 | 1 | illumination change; non-relevant static object |
| MEETING | 12 615 | 6 | 5 | dynamic occlusions; non-relevant static object |
| COFFEE | 2 437 | 4 | 3 | non-relevant static object (sitting person) |
| TABLE | 2 643 | 2 | 2 | — |
| CORRIDOR | 2 947 | 5 | 3 | illumination; dynamic occlusions; **small-sized object** |
| TWO DOORS | 2 460 | 4 | 2 | illumination/saturation changes |

**Methodology — two independent cue pipelines, mutually validated:**
1. **Depth pipeline.** Disparity background model by **running average that also tracks pixel validity**, with a deliberately slow adaptation rate. Background and current disparity are converted to metric depth (Z=fB/d etc.) and represented as **octree voxel grids**; recursive octree comparison yields **geometric scene changes** (this handles differing size/resolution/density/point-ordering between the two octrees). Changes are **re-projected into 2-D image space** for cheap aggregation and segmentation. Temporal association via an accumulator using (a) the fitted rectangle centre and (b) **area ratio R = Area_S/Area_N**; matched entries increment, occlusion/removal decrements. After **N = 5** observations, a **convex-hull volume estimate** from the depth values filters out too-small candidates.
2. **Intensity pipeline.** Zivkovic adaptive GMM background + **Porikli's dual background model** for stationary candidates. Deliberately **high-sensitivity / high-recall** (catches low-contrast, small objects) at the cost of many FPs — because fusion will clean them up.
3. **Fusion:** pairwise **IoU** of depth-box and intensity-box; **r > 50 % ⇒ match**. One-to-many allowed.
4. **Validation (two mechanisms):**
   - **Motion History** — running average of inter-frame differences of **gradient norms**, with slow integration. Transient objects don't accumulate; **quasi-stationary objects (standing/sitting people, moving vegetation) produce a marked signature** and are rejected above threshold **T_mh**. *This is a direct answer to the sitting-person problem that broke GE's PETS 2006 system.*
   - **Weakly-parametric region growing with Maximum Stability** — region growing from a **3×3 grid of seeds** inside the proposal, with the similarity threshold swept by Δ; stability `S(Rᵍ) = (|Rᵍ⁻ᐞ| − |Rᵍ⁺ᐞ|)/|Rᵍ|`; the first local minimum is a **maximally stable segment**. If any seed yields a stable segment contained in the proposal box, accept. This rejects proposals **lacking any boundary, structure or texture** — i.e. highlights and underexposed patches.
5. **Alarm:** validated proposal persisting **25 s**.

**Results:**

| Metric | DOOR | MEETING | COFFEE | TABLE | CORRIDOR | TWO DOORS |
|---|---|---|---|---|---|---|
| **Precision** | 1.00 | 0.71 | 0.75 | 1.00 | 0.60 | 0.50 |
| **Recall** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

**100 % recall everywhere**; precision degrades to 0.5–0.6 from two specific causes:
- **Transferred objects** — a pushed rolling chair, an opening/closing door: they *are* static objects, and further reasoning (full scene-object tracking, or geometric attributes like size/height/compactness) would be needed to reject them.
- **Highlights** — in TWO DOORS, opening/closing doors cause a sudden appearance of *valid disparity pixels*, read as a change in scene geometry.

**Speed: 5 fps full pipeline; 10 fps for stereo only.** Coverage: **~10 m × 10 m**, detects a **small backpack at up to 10 m**.

**Honest limitation:** evidence is accumulated in the back-projected 2-D image space, so **long occlusions decay a candidate away**. They propose accumulating proposals in 3-D instead.

---

#### **[P13] Using RGB-D Sensors for the Detection of Abandoned Luggage**
**Ajami & Lang** — Fraunhofer IPK / TU Berlin, Germany

**Sensor:** **ASUS Xtion PRO LIVE** (IR + adaptive depth + RGB), via OpenNI.

**Datasets:** 🔒 8 OpenNI (`.ONI`) recordings with synchronised RGB + raw depth, taken **inside a fully functional tram** at **2 camera locations**, covering **4 scenarios** (S1–S4) varying whether the object is displaced and whether the camera view is obstructed. They note the ONI format + niche purpose made comparison with other work impossible, and offer it as a reference point. Part of the German **InREAKT** project.

**Methodology:**
1. **Object segmentation with a three-background-model scheme, run on *both* RGB and depth:**
   - Initial background = **per-pixel median over 50 frames**.
   - **Median model** — continues, per-pixel circular buffer, median updated each frame (expensive but effective).
   - **Secondary model** — running Gaussian average with a **very low α**, updating **only pixels currently classified background** ⇒ prevents motionless objects from blending in.
   - **Primary model** — running Gaussian average, adapts faster than secondary, slower than median; **foreground areas adapt with lower α**. Best current-situation estimate.
   - **Model arbitration:** compare each model's histogram to the current frame's using OpenCV `compareHist` with **CV_COMP_CORREL**; the closest-to-reality model **overwrites** the primary. This coordinated update is the paper's core engineering idea.
   - Difference image cleaned with **threshold-with-hysteresis** (argued more accurate than morphological ops).
   - **Fusion:** `fg_merged = fg_rgb ∨ fg_depth`. Depth is resilient to illumination; RGB is sensitive. **Sudden-light-change detection:** if the segmented areas from RGB and depth differ by **> 20 %**, the background model is updated and the current RGB background estimate is discarded.
   - Size filter (`minEnclosingCircle`) + person rejection via OpenNI `userMap`; too-small or human regions are discarded **before** background adaptation, so misclassification doesn't poison the models.
2. **Feature detection & matching:** each candidate ROI is described with **SURF** (scale- and rotation-invariant) and stored with its centroid + depth. New candidates are matched against the database by SURF to avoid duplicates. Per frame, SURF re-verifies each stored object.
   - **Main dwelling counter** increments while a matched object's centroid hasn't moved; crossing the threshold ⇒ abandoned.
   - **Hidden dwelling counter** handles occlusion: if the stored **depth value at the centroid differs** from the current frame's (a smaller depth means something is in front), the object is occluded, and the hidden counter is later folded into the main counter. *Depth gives an unambiguous occlusion test.*
   - If the object is genuinely removed, the main counter **decreases twice as fast as it increases**; below a threshold the object is deleted.

**Results:** dwell threshold set to **40 s**; measured true-positive detection times:

| Location | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| One | 45 s | 52 s | 120 s | 140 s |
| Two | 43 s | 48 s | 106 s | 120 s |

**False-positive rate 0 %** — but they immediately caveat it: *"traced back to the limited amount of evaluation videos and the limitation of the number of actors inside a scene to two persons."* Behaviour in a fully packed tram was untested, justified by a project finding that *critical situations occur in a relatively empty tram*.

**Privacy note worth flagging:** the InREAKT design deliberately emphasises the detected object **without analysing personal data of anyone in the scene**, for data-protection reasons. Relevant if you're writing a deployment/ethics section.

---

#### **[P14] Usage of Optical Correlator in Video Surveillance System for Abandoned Luggage**
**Solus, Ovseník & Turán** — Technical University of Košice, Slovakia · **IEEE Informatics 2017**, pp. 349–352

**The outlier of the corpus — an optical-hardware approach.**

**Hardware:** **Cambridge optical correlator**, a **1/f Phase-Only Joint Transform Correlator (JTC)** — the "1/f" meaning a single optical Fourier transform stage reused twice. Input + reference images are displayed together on a **Spatial Light Modulator**, Fourier-transformed optically, and a non-linear camera captures the intensity → **Joint Power Spectrum**; the JPS is binarised/thresholded and fed back through the transform; the output correlation plane contains **correlation peaks per match**. System = IP cameras + correlator + server (pre-processing, reference database, correlator API, output evaluation, alerting).

**Datasets:** 🔒 tiny and synthetic-ish — **4 sets × 5 frames**, containing 3 luggage types and 2 people; in each frame the luggage stays put and the person moves progressively away.

**Pre-processing chain (C#):** median filter → gamma correction → colour filtration (mainly black) → greyscale → **Sobel edge detection** → blob filtration below minimum size → fill remaining blobs white.

**Method:** correlate frame 1 against frames 1..5; track the **change in correlation-peak coordinates (Δx)** between the peak representing the luggage and the peak representing the person. If Δx exceeds a preset value → alert.

**Results (Δx per frame pair, 4 experiments):**

| Frames | Exp1 | Exp2 | Exp3 | Exp4 |
|---|---|---|---|---|
| 1–1 | 7 | 7 | 7 | 7 |
| 1–2 | 7 | 7 | 8 | 14 |
| 1–3 | 27 | 25 | 31 | 34 |
| 1–4 | 49 | 45 | 52 | 49 |
| 1–5 | 67 | 70 | 77 | 71 |

Monotonic separation is demonstrated; also confirms peak displacement mirrors direction of motion.

**Assessment:** proof-of-concept only — no public dataset, no precision/recall, no crowd, 5-frame sequences. Its value is as a **hardware alternative** (optical correlation offloads the matching to physics) and as a reminder that the corpus contains non-CNN alternatives. Funded by KEGA 023TUKE-4/2017 and VEGA 1/0772/17.

---

#### **[P15] An Edge-Based Method for Effective Abandoned Luggage Detection in Complex Surveillance Videos**
**Dahi, Chikr El Mezouar, Taleb & Elbahri** — RCAM Laboratory, Djillali Liabes University, Sidi Bel-Abbes, Algeria · **Computer Vision and Image Understanding 158:141–151, 2017**

**Datasets — the broadest evaluation of any classical method here: PETS 2006 (camera 3) · PETS 2007 (all, focus on the hardest camera 1 of S8) · i-LIDS AVSS 2007 (Easy/Medium/Hard) · CDnet 2014 · ABODA.**

**Motivation:** *edges are more robust to illumination changes than pixel intensities, require no shadow removal, and describe the scene better.* Their explicit target is **precision** (killing false alarms) while keeping recall at 1.

**Methodology — edges in *both* the detection and the classification step (this is the delta vs. Kim et al. 2014, who used edges only for detection):**

1. **Stable-edge detection.** Edge-based background subtraction (Gruenwedel et al.) operating **independently in X and Y**: Sobel first derivatives → running-average background gradient per direction `B_{x,t} = B_{x,t−1} + α·D_{x,t}`, difference `D_{x,t} = G_{x,t} − B_{x,t}`, binary mask by **hysteresis thresholding** `F_{x,t} = hyst(|D_x|, T_low, T_high)`, then `F = F_x ∨ F_y`. **Simplification vs. the original:** they use a single background model, not short-term + long-term.
   - **Temporal accumulator per edge pixel:** `ACC += 1` if edge and `i mod 10 == 0`; `ACC −= 1` if not edge and `i mod 10 == 0`. **The `i mod 10` gate is deliberate** — updating every 10th frame prevents temporarily-static and slow-moving objects from accumulating.
   - `SEMask = hyst(ACC, AO_time/2, AO_time)`. Hysteresis (rather than a single threshold) is what survives **partial occlusion and slow movers**.
   - Stable gradients masked by SEMask, then **non-maximum suppression** for thin edges.
2. **Edge clustering into candidate boxes.** Each edge segment is boxed; boxes are grouped by a **recursive label-propagation algorithm** (Algorithm 1) using **both spatial distance D_th (minimum distance between the four corners, not centroids) and temporal distance T_th (difference in stability time)**. Using T_th means **two objects dropped at different times, even overlapping, get separate boxes.**
3. **Classification — two scores, replacing the usual "one score per failure mode" design.** Their framing: instead of separately testing for ghosts, illumination and still persons, just ask *"is there an object inside this box?"*, inspired by category-independent objectness (Edge Boxes / BING).
   - **Objectness score S_b.** Divide the box into Left/Right/Top/Bot regions. Accumulate edge-group lengths whose **mean gradient orientation** matches the expected boundary orientation for that region (`|sin²θᵢ − 1| < σ` for Top/Bot; `|sin²θᵢ| < σ` for Left/Right). Then
     `S_b = λ / ((L_reg − BB_W)(R_reg − BB_W)(T_reg − BB_L)(B_reg − BB_L))²`
     i.e. high when the accumulated boundary length in each region approaches the corresponding box dimension — a **convexity/closed-boundary test**. They note the original Edge Boxes objectness fails on small, complex surveillance objects, hence the redesign.
   - **Staticness score C_b = |φ(i)|/2**, where φ(i) is the set of inter-edge connections of edge group i. Assumption: a simple-boundary object has **≥ 2 connections per edge group**; fewer penalises the score. **Purpose: reject still persons**, who make small internal movements that fragment the accumulated edges (edges are highly motion-sensitive).
   - Only edge distributions with accumulation **> AO_time/2** are scored, so occluding moving objects don't contaminate.
   - Accept if `S_b > T1` **and** `C_b > T2`.

**Parameters (Table 1):** α = 0.005, T_low = 40, T_high = 70, σ = 0.5 (⇒ orientation deviation < π/4), λ = 10⁸, **T1 = T2 = 10⁻⁵**. Thresholds were tuned once on AVSS 2007 (five configurations swept; T1=T2=10⁻⁵ gave precision = recall = 1.0) and then **used unchanged on every other dataset**.

**Results:**

| Method | PETS 2006 R / P / F | AVSS 2007 R / P / F |
|---|---|---|
| Tian et al. 2011 | 1.0 / 0.85 / 0.92 | 1.0 / **0.35** / 0.52 |
| Pan et al. 2011 | — | 1.0 / 1.0 / 1.0 |
| Szwoch 2016 | 0.86 / 1.0 / 0.98 | 1.0 / 1.0 / 1.0 |
| Fan et al. 2013 | 0.80 / 0.95 / 0.87 | 1.0 / 0.97 / 0.98 |
| Lin et al. 2015 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| **Proposed** | **1.0 / 1.0 / 1.0** | **1.0 / 1.0 / 1.0** |

⚠️ **Important fairness note they make themselves:** Lin et al. also achieve F = 1.0, **but restricted their detection area** (to the platform in AVSS 2007 and the waiting zone in PETS 2006). *This method runs over the whole scene.*

**ABODA:** matches or beats Lin and Wahyono on most videos. **No false alarm in video 5** (where Lin had 1); **fails video 11** (crowded, object too small) where Lin got the TP but with 3 FPs. Videos 7–8 (light switching) each cost them 2 FPs.

**Processing time:**

| Module | 720×576 | 320×240 |
|---|---|---|
| Stable edge detection | 20 ms | 4 ms |
| Clustering | 2.4 ms | 0.7 ms |
| Orientation extraction | **32 ms** | 4.5 ms |
| **Overall** | **54.4 ms (18 fps)** | **9.2 ms (108 fps)** |

vs. **Szwoch 2016: 49 fps** and **Lin 2015: 29 fps** at 320×240. C++/OpenCV on an i7 laptop.

**Deep comparison vs. GMM (Tian et al.)** — worth reading if you're choosing a foreground model. Their three arguments: (1) in a cluttered airport with slow crowds, similarly-coloured objects crossing the same region **push the foreground Gaussian into the background set**, producing a noisy static mask; edge-level temporal accumulation avoids this because *"the probability that contours of two moving objects will overlap is very low"*; (2) GMM's second Gaussian is designed for **repetitive** background motion (waving trees), so it usually models periodic motion rather than static objects; (3) **different pixels of a newly-static region reach the second Gaussian at different speeds** under partial occlusion or colour similarity → fragmented static regions. Their hysteresis thresholding solves (3) directly.

**Stated future work:** add an efficient detection + tracking module *for the owner* — which they never did.

---
### 3.3 Era 3 — Deep learning arrives (2018–2020): CNNs bolted onto background subtraction

---

#### **[P16] Application of YOLO Deep Learning Model for Real Time Abandoned Baggage Detection**
**Santad, Silapasupphakornwong, Choensawat & Sookhanaphibarn** — BU-Multimedia Intelligent Technology Lab, Bangkok University, Thailand · **IEEE GCCE 2018**, pp. 157–158 *(2-page short paper)*

**Historical significance:** among the first to argue that **background subtraction should be replaced, not augmented**. Their critique: BS methods *"are not invariant with camera plane, only detect static and specific objects and area"*, and most systems require the camera to be adjusted to eye level before analysis, and break under irregular lighting.

**Methodology:** every frame → **YOLO** → list of detected objects. Compare each detection with the previous frame's list; new person/object → instantiate a **tracker class**; existing objects compared against tracker data to analyse movement. **If a person walks away from their baggage beyond an administrator-set threshold distance → alarm.** Events logged to file for later crime investigation. A GUI exposes the parameters (original video, detection overlay, control console, movement-history tracking, detection statistics, event log) — the GUI is the point: **the parameters are what make the system invariant to lighting and camera position.**

**Datasets:** 🔒 **11 self-recorded conditions** (iPhone on a tripod as the surveillance camera), a genuinely well-designed factorial:

| Case | Light source | Camera position | Walk direction |
|---|---|---|---|
| 1 | Outdoor (sun) | High | 90° |
| 2 | Outdoor | High | 0° |
| 3 | Outdoor | Normal (eye level) | 90° |
| 4 | Indoor fluorescent | High | 45° |
| 5 | Indoor fluorescent | Normal | 45° |
| 6 | Indoor window light | High | 0° |
| 7 | Indoor window light | Normal | 0° |
| 8 | Indoor blue light | High | 90° |
| 9 | Indoor blue light | Normal | 90° |
| 10 | **Sky Train station** | Normal | 45° |
| 11 | **Subway station** | Normal | 45° |

Four lighting types (sunlight, fluorescent, high-dynamic-range window light, coloured/blue fluorescent), two camera heights (ceiling vs. eye level), three walk directions relative to the camera plane. Subject carries a backpack, drops it, walks out.

**Results:** *"Our system had achieved in detected the person and his bag and alarm when he abandoned it for all 11 cases."* No precision/recall/timing. Take it as a feasibility demonstration, but **the experimental design is the reusable part** — it's the cleanest lighting × geometry factorial in the corpus.

---

#### **[P17] Intelligent Surveillance System for Abandoned Luggage** *(Terk Edilmiş Bagaj Algılama için Akıllı Güvenlik Sistemi)*
**Altunay, Karademir, Topçu & Direkoğlu** — Middle East Technical University, Northern Cyprus Campus · **SIU 2018** *(paper is in Turkish with an English abstract)*

**Methodology — a clean hybrid: classical BS proposes, Faster R-CNN disposes.**
1. **Background subtraction + dual foreground modelling.** MoG background model; **long-term and short-term** foregrounds, the short-term updated more often; the difference between them yields **newly-static objects**. Published parameters: history = 1 for both; pixel threshold **120 (short-term)** and **1000 (long-term)**; learning rate **0.002 (short-term)**, **0.0001 (long-term)**.
2. **Luggage recognition + owner association** with **Faster R-CNN**. Trained on a combined person + bag dataset via the TensorFlow Object Detection API, starting from a **Faster R-CNN ResNet model pretrained on COCO**, on an NVIDIA Quadro GPU.
   - **A detector-choice ablation** (Table 1) comparing YOLO vs Faster R-CNN on part of their test set, by detection time (NTP, seconds) and accuracy (DY):

| Bags in image | Faster R-CNN time / acc | YOLO time / acc |
|---|---|---|
| 1 | 1.14 s / 99 % | 0.36 s / 68 % |
| 1 | 1.30 s / 99 % | 0.38 s / 47 % |
| 1 | 1.17 s / 98 % | 0.37 s / 61 % |
| 1 | 0.96 s / 99 % | 0.32 s / 31 %+56 % (1 FA) |
| 2 | 1.12 s / 99 %+93 % | 0.43 s / 79 %+7 %+1 % (1 FA) |
| 2 | 1.35 s / 99 %+95 % | 0.36 s / 36 %+11 % |
| >5 | 0.96 s / 89–99 % | 0.32 s / 42 % |
| **Avg** | **1.14 s / 97.43 %** | **0.36 s / 52 %** |

   They chose **Faster R-CNN despite being ~3× slower**, explicitly to reduce false alarms and preserve reliability. (Note this is 2018-era YOLO — later papers reverse the conclusion.)
   - **Efficiency trick:** they pass the *region around* the static object to Faster R-CNN, not the object itself, because occlusion/shadow/brightness shift the box over time. Duplicate boxes representing the same object inside that region are **ignored**, so recognition runs **once per object**, not per frame.
   - **Owner** = nearest person found by the same model. **Metric scale without calibration:** a standard human height is registered to the person's bounding box, giving a **pixels→centimetres conversion** at that image location; the bird's-eye view of the environment is proportioned to the camera's field of view.
3. **Event analysis:** boundary radius **3 m** (taken from the PETS 2006 measurements); when the owner exits it, a **30 s countdown** starts; if they don't reappear inside 3 m → alarm.

**Datasets:**
- 🔒 **Custom bag dataset: 3 000 images**, deliberately chosen as **everyday scenes rather than plain single-colour backgrounds**, so learned bags match reality.
- **INRIA Person** for the person class.
- **ABODA** — indoor videos only (matching their stated scope).
- 🔒 **Custom 5-video test set**.

**Results:**

*ABODA (indoor subset):* **Precision 83.33 %, Recall 100 %** — better than Lin et al. 2015, who created the dataset.

| Video | GT | TP | FP | Scenario |
|---|---|---|---|---|
| 1 | 1 | 1 | 0 | crowded corridor |
| 7 | 1 | 1 | 0 | indoor, variable light |
| 8 | 1 | 1 | **1** | indoor, variable light |
| 9 | 1 | 1 | 0 | indoor |
| 10 | 1 | 1 | 0 | indoor |

*Custom video set:* **Precision = Recall = 87.5 %**

| Video | GT | TP | FP | FN | Scenario |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 0 | 0 | lab |
| 2 | 2 | 2 | **1** | 0 | crowded lab |
| 3 | 1 | 1 | 0 | 0 | indoor |
| 4 | 2 | 2 | 0 | 0 | crowded indoor |
| 5 | 2 | 1 | 0 | **1** | classroom, variable light |

**Stated next step:** use **person re-identification** to find the owner by examining past scenes — a theme that recurs in Dogariu and Melebari.

---

#### **[P18] Real-Time Deep Learning Method for Abandoned Luggage Detection in Video**
**Smeureanu & Ionescu** — University of Bucharest / SecurifAI, Romania · **EUSIPCO 2018**, pp. 1775–1779 *(arXiv:1803.01160 is the same paper — the two PDFs in your folder are duplicates)*

**Claim:** *"To our best knowledge, we are the first to train a **cascade of convolutional neural networks** for abandoned luggage recognition."*

**Methodology — two stages.**

*Stage 1 — Static Object Detection (SOD), 3 steps:*
- **A. Foreground** by standard background subtraction (estimated background subtracted from each frame), then **erosion + dilation**. Contains both static and moving objects.
- **B. Motion** by subtracting frames **5 frames apart** → contour of moving objects; erosion + dilation to fill; connected components; **convex hull** per component. Result: a motion mask of moving objects as convex blobs.
- **C. Static pixels** = foreground mask **−** motion mask → connected components → bounding boxes → sub-images. Static objects are **tracked across frames by IoU > 0.5**.

*Stage 2 — Cascade of CNNs (CCNN), both **GoogLeNet** pretrained on ILSVRC with **only the last layer retrained**:*
- **CNN 1:** luggage vs. other objects.
- **CNN 2:** **abandoned** luggage vs. **attended** luggage (positives: abandoned; negatives: luggage with people standing by). Applied only to CNN-1 positives.
- **Crucial detail:** before CNN 2, the box of size h×w is **expanded to 2h × 3w** so that nearby people are actually inside the crop — the network can only judge "attended" if it can see the attendant.
- Applied every **10 frames** for real time; track scores **temporally smoothed with a 25-frame (1 s) Gaussian**, sign transfer to labels, then **majority voting** per track.

*Training data — the most reusable idea in the paper:*
- **Internet-collected images:** **2 207 abandoned luggage · 2 000 attended luggage · 8 035 other objects** (people, cars, buses, trains…). 80/20 split; training set augmented with flipped and blurred versions.
- **Synthetically generated scene-specific samples:** superimpose **template luggage items at random locations over the estimated background of each individual scene** for positives, random background sub-images for negatives; for CNN 2, superimpose **people carrying/standing by luggage**. Motivation: collecting real per-scene samples is too slow for fast deployment.

**Preliminary classification results:**

| | Precision | Recall | Accuracy |
|---|---|---|---|
| First CNN | 97.31 % | 82.12 % | 96.37 % |
| Second CNN | 96.96 % | 94.11 % | 95.36 % |

**Datasets (4, all public): AVSS 2007 · PETS 2006 (camera 3) · PETS 2007 (S7, S8, camera 3) · TCD.** **14 test videos, 42 869 frames total.** They **manually annotated all videos with ground-truth boxes**, since none provide them.

**Metrics:** frame-level and pixel-level **precision / recall / F1**. Pixel-level counts a detection correct at **IoU > 0.2** (a low bar — flag this). Frame-level counts a frame correct if it contains ≥ 1 abandoned item, with no overlap requirement.

**Results:**

| Dataset | Method | Frame P / R / F1 | Pixel P / R / F1 |
|---|---|---|---|
| **AVSS 2007** | SOD+CNN (baseline) | 60.17 / 60.87 / 60.52 | 41.19 / 53.03 / 46.37 |
| | SOD+CCNN | 97.77 / 51.89 / 67.80 | 97.77 / 51.89 / 67.80 |
| | **SOD+CCNN+Generated** | **97.48 / 66.59 / 79.13** | **97.47 / 65.70 / 78.49** |
| **PETS 2006** | baseline | 68.01 / 69.54 / 68.77 | 68.00 / 69.54 / 68.76 |
| | SOD+CCNN | 83.25 / 69.54 / 75.78 | 83.25 / 69.54 / 75.78 |
| | **+Generated** | **95.67 / 83.74 / 89.31** | **95.67 / 83.74 / 89.31** |
| **PETS 2007** | baseline | 65.35 / 99.61 / 78.92 | 65.17 / 99.61 / 78.79 |
| | SOD+CCNN | 69.13 / 99.61 / 81.62 | 68.99 / 99.61 / 81.52 |
| | **+Generated** | **97.47 / 99.61 / 98.53** | **97.46 / 99.61 / 98.52** |
| **TCD** | all three | 98.62 / 100 / 99.31 | 98.62 / 100 / 99.31 |
| **Overall avg** | baseline | 70.32 / 76.33 / 73.20 | 66.22 / 74.65 / 70.18 |
| | SOD+CCNN | 86.54 / 74.41 / 80.02 | 86.52 / 74.40 / 80.00 |
| | **+Generated** | **96.74 / 84.65 / 90.29** | **96.73 / 84.46 / 90.18** |

**The two headline deltas:** the **cascade** buys ~+7 F1 over a single CNN; **scene-specific synthetic samples** buy a further ~+10 F1, and roughly **+20 F1 over the baseline** overall. Precision jumps from 70 % → 97 %.

**Speed: ~40 fps** on an **Intel Xeon E5 1.7 GHz CPU with 32 GB RAM, no parallel threading.**

---

#### **[P19] Abandoned Object Identification and Detection System for Railways in India**
**Arora, Dhar, Singh & Mishra** — Amity University Uttar Pradesh, India

**Motivation:** India operates the world's largest railway system; both malicious and accidental abandonment need tracking.

**Methodology:** classical, three steps.
1. **Background subtraction + static-region identification** using a **mixture of Gaussians**, with distinct thresholds for the moving-object foreground mask and the static-region mask. **Texture information** used to remove shadows and cope with rapid lighting change. Two frames are compared against the background to confirm the object really is motionless.
2. **Object-sort discovery** — distinguishing abandoned objects from **removed objects (ghosts/false detections)** via a static-region segmentation strategy, which they claim beats prior edge-based procedures.
3. **Abandoned-object recognition + feature extraction.** Dwell window **85–100 frames** for the PETS 2006 data. Box drawn around the object; red until the bag is retrieved; the decision is taken over several frames to build confidence. **Feature extraction to a `.csv`**: **bag colour, date, time, frame number**, plus a saved screenshot for owner association. Alarm persists until the owner returns or a supervisor resets it. Capacity: **up to 100 simultaneous object profiles**; **single camera view only**.

**Dataset:** **PETS 2006.**

**Results:**

| Dataset | Alarm count (frames) | Detected? | Objects detected |
|---|---|---|---|
| S1 | 85 | Yes | 1 |
| S2 | 85 | Yes | 1 |
| S3 | 85 | Yes | 1 |
| S4 | 95 | Yes | 1 |
| S6 | 85 | Yes | 1 |
| **S7** | 85 | **No** | — |

**6 sequences tested, 5 detected, S7 (the 5-star crowded one) missed.** Note S3 registering a detection is arguably a false positive under PETS rules (the owner never leaves).

**Extra value:** the paper contains a **12-row literature summary table** (method / advantages / limitations / dataset / accuracy) covering Backpack, motion-based recognition, homographies+heuristics, temporal templates, region-matching, geometric shape models, planar homography, temporal-flow-of-events, optical-flow motion models, pose-preserving dynamic shape models and Lagrangian dynamics. Useful for a related-work section — though most of its "Accuracy" cells read *Not Available*, which is itself a finding about the field.

**Weakness:** the reported "alarm count" of 85 frames is a *dwell threshold*, not a performance metric; no precision/recall/timing error is given.

---

#### **[P20] Human-Object Interaction: Application to Abandoned Luggage Detection in Video Surveillance Scenarios**
**Dogariu, Ştefan, Constantin & Ionescu** — University Politehnica of Bucharest · **COMM 2020**, pp. 157–160
*(Your folder contains this paper twice — `Human-Object_Interaction_...pdf` and `Dogariu_COMM_2020.pdf` are the same paper.)*

**Claim:** *"To the best of our knowledge, this is the first system to perform all these actions in an end-to-end system"* — i.e. detect the abandoned bag **and** identify who left it **and** track that person across the whole camera network. The Boston Marathon bombing (4+ days to identify suspects) is their stated motivation.

**Methodology — one network, three jobs.** Everything is built on **Mask R-CNN**, and critically **the same RPN feature vectors drive all three modules**, so re-identification is *almost free*.

1. **Unattended baggage detection.** Only 4 COCO classes retained: `person`, `backpack`, `handbag`, `suitcase` — the last three grouped as **`baggage`**. **Unattended ⇔ the baggage bounding box does not intersect any detected person's bounding box.** Deliberately crude; they call it "a good compromise for the problem at hand." **Detection threshold lowered to 0.5** (biasing toward low false negatives, since missing a bag is the costly error); NMS kept at 0.7.
2. **Suspect detection.** Take the Mask R-CNN **feature vector** of the abandoned bag; search all available images for the same bag, ranking by **Euclidean distance** `d(f_q, f_x) = √Σ(f_q(i) − f_x(i))²`, **conditioned on a person's box intersecting the bag's box**. Top-ranked image ⇒ the person in it is the suspect.
3. **Suspect re-identification.** Same procedure applied to the person, but ranked **per camera**, so each camera contributes its best sighting — the point is to trace the suspect's **path** through the surveilled perimeter, not to return several shots from one camera.
   - **Scale invariance is the neat bit:** a person can be 30×60 px on one camera and 210×500 px on another; because comparison is on **fixed-length feature vectors**, the size difference is irrelevant.

**Datasets:** **MS-COCO 2017** (~330 k images, 200 k labelled, 80 classes) for detection; **CUHK03** for re-ID; 🔒 a demo set of **1 hour of the research centre's own CCTV**, downsampled to **1 fps** and motion-triggered → only **120 images**, covering basement, ground floor and building exterior, with a staged backpack abandonment plus several decoy people carrying backpacks.

**Results — backbone selection (COCO, bbox AP@IoU=0.75, single NVIDIA Quadro M4000):**

| Backbone | AP@0.75 | Inference (s/img) |
|---|---|---|
| R50-C4 1x | 35.7 | 0.392 |
| R50-DC5 1x | 37.3 | 0.408 |
| **R50-FPN 1x** | 37.9 | **0.228** |
| R50-C4 3x | 38.4 | 0.398 |
| R50-DC5 3x | 39.0 | 0.396 |
| **R50-FPN 3x** ← chosen | **40.2** | **0.231** |
| R101-C4 3x | 41.1 | 0.482 |
| R101-DC5 3x | 40.6 | 0.474 |
| R101-FPN 3x | 42.0 | 0.308 |
| X101-FPN 3x | **43.0** | 0.591 |

**Their reasoning is worth quoting:** *"it is better to opt for a model which sacrifices a part of the detection accuracy in favor of a faster inference time. The detection accuracy loss can be overcome by setting a lower detection threshold to force additional proposals and decrease the false negative rate. Decreasing inference time is, however, far more difficult."*

**Re-ID:** **top-1 accuracy 70.8 %** on CUHK03. They also tried an Inception backbone (per Xiao et al.) but rejected it — more parameters, slower inference.

**Qualitative demo result:** successfully detected the abandoned bag, identified the person who left it, and located them on individual cameras — including recovering the moments the person **entered the building carrying the backpack** and **left without it**, which narrows the investigation window, and surfacing **who the suspect interacted with**. A demo GUI was built for operators.

**Assessment:** no quantitative abandonment metrics (no precision/recall on any abandonment benchmark), and the "no bounding-box intersection" rule is far weaker than a calibrated metric radius. Its real contribution is the **forensic/retrieval framing** — detection is only the first of three tasks.

---
### 3.4 Era 4 — Deep learning matures (2022–2024): purpose-built architectures

---

#### **[P21] HLDNet: Abandoned Object Detection Using Hand Luggage Detection Network**
**Dohun Kim, Heegwang Kim, Yeongheon Mok & Joonki Paik** — Chung-Ang University, Seoul · **IEEE Consumer Electronics Magazine 11(4):45–56, July 2022**

**The most conceptually distinctive deep-learning paper in the corpus.** Its premise: **don't detect the abandoned object — detect the *hand luggage* and the act of dumping it.**

Their reasoning: an abandoned object is by definition *dumped by a human hand*. Background subtraction has a **fundamental** false-detection problem (flickering neon signs, static objects, PTZ cameras, illumination change, camera shake), and *"although it is difficult to detect an unspecified abandoned object using a learning-based method, it is possible to detect an object carried by a human hand."*

**Methodology — three parts:**

1. **Pedestrian detection and tracking.** **RetinaNet** detector (retrained with on-site CCTV data) + **KCF (Kernelized Correlation Filter)** tracker. KCF chosen for two reasons: correlation replaced by element-wise operations in the Fourier domain (fast), and online training (robust to shape change). The detector re-runs at intervals so a failed track can be re-acquired; the same pedestrian is re-tracked by **hue-channel histogram similarity**.

2. **HLD Network — the contribution.** A **merged network** combining:
   - **OpenPose** (VGG-19 backbone, pretrained on COCO) for **keypoint detection** — confidence maps for joints + part affinity fields to connect them, yielding the **hand location**.
   - **SSD** for generic object detection — 6 feature maps at **38×38, 19×19, 10×10, 5×5, 3×3, 1×1**, concatenated into one multibox layer with **8 732 boxes**.
   - **The fusion:** instead of running NMS straight after concatenation, they insert a **Gaussian filter operation layer**. A Gaussian centred on the detected hand reweights every detection's confidence: **C_G = G(x,y)·C(x,y)**. Detections near the hand keep their confidence; distant ones converge to zero. A Gaussian is **also applied at the elbow** to cover cases where the luggage occludes the hand or the hand isn't detected.
   - **Why this beats naive SSD:** the two obvious SSD strategies each fail — "nearest object to the hand" fails with clutter around the hand; "highest confidence object" fails when something else scores higher. The Gaussian **combines both cues** in one operation.
   - **Training data:** SSD trained on **COCO trainval35k + Pascal VOC 2007+2012 (21 classes)**; then, because "unspecified abandoned object" can't be a class, they **trained all objects located in or right below the hand as a single class** using the **KISA** and 🔒 **in-house Chung-Ang University (CAU)** datasets (plastic bags, bags, paper boxes).
   - **Hyperparameters:** input resized 300×300; VGG-16 + 5 extra layers; **batch 16, LR 10⁻³, momentum 0.9, weight decay 5×10⁻⁴, 120 000 iterations ≈ 21 h on an RTX 2080Ti**. Loss = SSD standard (softmax confidence + smooth-L1 box regression), positive:negative = 1:3, IoU match threshold 0.5.
   - **Portability claim:** the Gaussian layer *"can be embedded in object detection networks such as YOLO and Faster R-CNN."*

3. **Abandoned-object decision.** Detected hand luggage is **not tracked immediately** — it's stored and compared to the next frame by **hue-histogram similarity**; only after **5 consecutive similar detections** is it accepted as hand luggage (this also serves as re-tracking when KCF fails). Then KCF tracks it and measures owner distance. **Rule: if D > 2 × W (owner's bounding-box width) sustained for 5 s → abandoned.** (Condition sourced from Luna et al.'s survey and verified experimentally.) *Note this is a completely calibration-free criterion.*

**Datasets:** **ABODA** (main evaluation) + **KISA** (outdoor stress test) + 🔒 CAU in-house.

**Results:**
- **ABODA: successfully detected abandoned objects in 10 of 11 videos** — matching the best reported (Park's and Shyam's). **Fails video 11** due to inaccurate keypoint detection on small objects plus occlusion in a crowd.
- **Their critique of Shyam's method is sharp and generalisable:** Shyam trained on **backpacks and handbags**, so it looks superb on ABODA *because most ABODA objects are backpacks or handbags* — *"in a real environment where abandoned objects exist as various objects, the performance is not promising."*
- **IoU results (the metric that actually validates HLDNet):**

| Video | Scenario | (a) IoU while carrying | (b) IoU over the 5 dumping frames |
|---|---|---|---|
| V1 | Outdoor | 0.6487 | 0.8186 |
| V2 | Outdoor | 0.8344 | 0.8903 |
| V3 | Outdoor | 0.8446 | 0.8788 |
| V4 | Outdoor | 0.8838 | 0.9398 |
| V5 | Night | 0.6838 | 0.8210 |
| V6 | Illumination change | 0.6782 | 0.8315 |
| V7 | Illumination change | 0.8479 | 0.8940 |
| V8 | Illumination change | 0.6617 | 0.8251 |
| V9 | Indoor | 0.8682 | 0.9372 |
| V10 | Indoor | 0.8872 | 0.9529 |
| V11 | **Crowded** | **—** | **—** |
| **Average** | | **0.7839** | **0.8789** |

IoU is lower while walking (occlusion, motion) than at the moment of dumping — and the dumping moment is the one that matters.

- **Direct dual-background comparison on KISA-style outdoor scenes (Table 3) — the money table:**

| Video | Scene | GT | Dual background | HLDNet |
|---|---|---|---|---|
| (a) | Illumination change | 0 | **3 FP** | **0** |
| (b) | Beach with waves | 0 | **2 FP** | **0** |
| (c) | Moving cars on a bridge | 0 | **2 FP** (stopped vehicle read as abandoned) | **0** |
| (d) | Small object | 1 | **0 (missed)** | **1** |

Seven false alarms and one miss for dual-background modelling, **zero errors for HLDNet**, on exactly the conditions that break background subtraction: waves periodically destroy the background region; moving vehicles prevent correct background formation; low light makes short- and long-term backgrounds diverge on tiny illumination changes; noise removal that suppresses false detections also removes genuinely small objects.

**Honest limitation:** *"the performance of abandoned object detection is significantly limited in a crowded situation with more than ten people, which is considered a chronic problem in the field."* Their rationalisation: *"the action of illegally abandoning an object usually occurs in a secluded place."*

**Bonus capability:** because the network detects hand luggage generally, it can also flag **dangerous weapons or restricted hand luggage**, and integrate with other abnormal-behaviour detectors (intrusion, loitering, violence).

**Funding:** Korean NRF (2020M3F6A1110350) and Civil-Military Technology Cooperation (19CM5119).

---

#### **[P22] Machine Learning Approaches for Abandoned Luggage Detection**
**Chaitra K M & Mustafa Basthikodi** — Sahyadri College of Engineering & Management / VTU, India · **IEEE DISCOVER 2023**, pp. 8–12

**Methodology:** a comparative study rather than a new architecture.
- **Models:** **CNN** (transfer learning, fine-tuning pretrained ResNet/VGG on ImageNet); **SVM**; **ensembles — Random Forest and Gradient Boosting**.
- **Features:** hand-crafted (**HOG, LBP, colour histograms**) combined with learned CNN features.
- **Architectures discussed:** single-stage (YOLO, SSD, needing careful anchor-box tuning) vs two-stage (Faster R-CNN, more accurate but slower).
- **Hyperparameter policy:** decaying LR schedules; **smaller batches for fine-tuning pretrained models, larger batches when training from scratch**; anchor boxes tuned to abandoned-luggage instance statistics; augmentation (rotation, brightness, scaling) tuned to avoid unrealistic artefacts.
- **Splits:** 70 % train / 15 % validation / 15 % test.
- **Hardware:** NVIDIA Tesla V100 GPUs, TensorFlow + PyTorch.
- Also frames the general surveillance pipeline: object segmentation → object classification → object tracking → action recognition.

**Datasets: ⚠️ read this carefully — they are *not* abandonment datasets.**
- **UMN** (Mehran et al. 2009): 11 clips consolidated into one 4:17 video, **7 739 frames**, 640×480 @ 30 fps, 1 indoor + 2 outdoor scenes, **temporal ground truth only**. Normal behaviour transitioning to abnormal.
- **CUHK Avenue** (Lu et al. 2013): **37 videos** (16 normal for training, 21 abnormal for testing), **30 652 frames** (15 328 train / 15 324 test), 640×360 @ 25 fps, **47 abnormal events** in three categories: strange actions (running, object tossing, loitering), wrong direction, **abnormal objects** (people carrying unusual items such as bicycles).

**Results:**

| Model | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| CNN | 0.92 | 0.87 | 0.89 | **0.94** |
| SVM | 0.84 | 0.79 | 0.81 | — |
| Ensemble (Random Forest) | 0.91 | 0.88 | 0.89 | — |
| **Ensemble (Gradient Boosting)** | **0.93** | **0.89** | **0.91** | — |

**Acknowledged limitation:** class imbalance — abandoned-luggage instances are rare, which forced a higher precision-recall trade-off.

> **⚠️ Assessment — be careful citing this one.** The benchmark datasets described (UMN, Avenue) are **crowd-anomaly / abnormal-event** datasets, not abandoned-luggage datasets, and neither contains annotated abandonment events. Neither PETS, AVSS nor ABODA is used. The reported numbers therefore do not measure abandoned-luggage detection in the sense the rest of this corpus means. Treat it as a **survey of ML options** with an indicative experiment, not as a comparable result.

---

#### **[P23] Research on Airport Baggage Anomaly Retention Detection Technology Based on Machine Vision, Edge Computing, and Blockchain**
**Chen, Mao, Yang, Du & Song** — Second Research Institute of CAAC, Chengdu / China Internet Network Information Center · **IET Blockchain 4:393–406, 2024**

**A genuinely different problem — worth including precisely because it isn't "abandoned luggage" in the security sense.** The target is **checked baggage that falls off the conveyor belt and gets stuck** in the airport baggage-handling system — an operations/safety problem, not a terrorism one. It complements **RFID** tracking, which cannot see what happens *between* two RFID scan points.

**Methodology — three layers:**
1. **Vision.** **YOLOv5** (CSPDarknet-style backbone, FPN neck, multi-scale detection heads with auto-adjusted anchor boxes, Leaky ReLU, **CIoU** loss).
2. **Anomaly-retention algorithm.** Step 1: inference gives luggage corner coordinates + class. Step 2: define the **ROI** = the trajectory area where luggage normally travels on the belt; monitor boundary-crossing behaviour; on a crossing, start frame counting; **if after a set number of frames a luggage target with the same classification is still outside the ROI ⇒ anomaly retention** → alarm with current frame, time, camera ID, luggage type ID.
   - **ROI-definition ablation (Table 2):** *Approach 1* = define the area **outside** the belt as ROI; *Approach 2* = the belt trajectory itself. Comparative experiment across **5 cameras at different locations**:

| | Approach 1 | Approach 2 |
|---|---|---|
| Average ROI area ratio | 62.4 % | **51.8 %** |
| Single-frame processing time | 52.3 ms | **44.6 ms** |

   Approach 1 produced a much larger ROI in spacious camera views, hurting tracking and costing **~7.7 ms/frame more** → **Approach 2 chosen**.
   - **Perspective-correction algorithm:** an **improved ray-casting method** correcting near-large/far-small distortion. From each of the luggage's four corner points **A, B, C, D**, rays are cast (one perpendicular and one parallel to the direction of luggage movement). The ROI polygon must have ≥ 3 vertices. For each corner the algorithm counts intersections between the ray and the polygon edges: **an odd count means that corner is inside the ROI**. **Abnormal retention is declared only when all four corners are outside the ROI** (luggage Y in their Fig. 5); otherwise the item is normal (luggage X). This is what stops luggage appearing disproportionately large or small at the camera angle from obscuring the ROI boundary.
3. **Edge + blockchain.** Multiple **NVIDIA Jetson AGX Xavier** edge servers on Ubuntu 18.04 form a **private chain**; alarm images are encrypted (a **chaos + DNA image encryption** scheme) and stored distributed; a **tamper-proof verification algorithm** lets data consumers verify authenticity. Stored elements: image data, anomalous-luggage type, timestamp, camera ID. **Hyperledger Fabric** in Docker, smart contracts in **Go**. Rationale: keep sensitive data local (privacy + latency), and make the alarm record immutable and auditable.

**Dataset:** 🔒 **> 4 600 original images**, annotated and augmented (geometric + pixel-level), **5 luggage classes**: hard case (0), cardboard box (1), cardboard box with packing (2), soft bag (3), soft bag with packing (4). ⚠️ **The training set contained no instances of luggage abnormal retention** — the detector learns luggage, the algorithm infers retention.

**Results — per-class detection accuracy (2 h live test at an airport):**

| Class | Accuracy |
|---|---|
| Hard case | 93.3 % |
| Cardboard box | 95.4 % |
| Cardboard box with framing | **97.6 %** |
| Soft package | **90.6 %** |
| Soft package + box | 92.7 % |
| **Average** | **93.9 %** |

*(Accuracy = total quantity detected in 2 h / total quantity of luggage in 2 h.)*

**System test:** 12-hour video stream with **artificial luggage retention created every half hour**; **Hikvision DS-2CD7A8XYZUV-WLS** cameras at **3840×2160 @ 25 fps**; 1000 Mbps network.

> ⚠️ **Metric transposition — now resolved.** The **abstract** and the **conclusion** agree: *"a luggage recognition rate of **96.9 %** and an anomaly detection rate of **95.8 %**."* **Section 5 states the reverse.** Two of three statements agree, so **96.9 % recognition / 95.8 % anomaly retention** is almost certainly the correct reading — but cite it with the caveat.

**System-level feature comparison (Table 5)** — against RFID tracking systems and a prior visual stranded-luggage system, theirs is the only one offering all of: *targeted at check-in scenarios · provides localization · real-time response · information-transmission security · information-storage security*.

**Their own stated shortcomings:** *"The current dataset is insufficient to cover all types of luggage for detection, and further efforts are needed to increase the dataset and optimize the model"*, and *"the validation is limited to a single airport."*

**Relevance to you:** the **ROI-crossing + frame-count** formulation is a much simpler and more reliable abandonment proxy than owner-distance **when the scene has a known "correct" region**. Also the only paper addressing **evidence integrity** — if your work touches deployment or legal admissibility, this is the citation.

---

#### **[P24] Enhanced Abandoned Object Detection through Adaptive Dual-Background Modeling and SAO-YOLO Integration**
**Lei Zhou & Jingke Xu** — Shenyang Jianzhu University, China · **Sensors 24(20):6572, 2024**

**Target problem stated precisely: small objects and occluded objects**, which drive false and missed detections.

**Methodology — a classical front-end and a purpose-built detector back-end.**

1. **Adaptive dual-background model.** Standard dual-background (long-term with low learning rate, short-term with high learning rate; their difference reveals static foreground). Their addition: **impact factors and an adaptive learning rate** built on an **Adaptive Gaussian Mixture Model**, so the update rate responds to lighting and target complexity, dynamically tuning noise sensitivity. Small/occluded objects are hit hardest by noise, hence the focus.
2. **Improved PFSM (Pixel-based Finite State Machine)** for extracting suspicious static foreground — **with an added occlusion state**.
3. **SAO-YOLO (Small Abandoned Object YOLO)**, built on a **YOLOv5** baseline:
   - **SAO-FPN** — feature-extraction network redesigned for small objects. **Multi-scale detection-layer selection is empirically justified** (see the O-level table below), plus **pruning of redundant modules** to cut information loss in forward propagation.
   - **SODHead** — a lightweight decoupled head. An embedded **LFEM (Local Feature Extraction Module)** takes the **lower detection layer's** output, applies **cropping, padding and rearranging** to pull out key local features, and fuses them into the **higher-layer** features via **self-attention**. Decoupling avoids mutual interference between classification and localisation information — the mechanism for handling occlusion.

**Adaptive learning rate — the actual equations.** Two phases: an **initial phase** with a high learning rate to converge the background model fast, then a **stable phase** governed by

- `α = λ₀` if `T < T₀`, else `α = λ₀(1 − γ)`
- `ε = N_obj / N_all` (**object-complexity factor** — the proportion of pixels detected as target; higher complexity ⇒ lower learning rate)
- `δ = 1 − H_{t−1}/H_t` (**lighting-change factor**, computed from **image entropy**; set to 1 on a significant lighting change to raise the learning rate and re-stabilise, back to 0 once stable)
- `γ = ε(1 + δ)` if `ε < 50 %`, else `γ = 0.5(1 + δ)`
- **λ₀ ∈ [0.03, 0.06] for the long-term model; λ₀ ∈ [0.1, 0.3] for the short-term model.**

**Improved PFSM — the state machine in detail.** Both background models are binarised (F_L, F_S) and each pixel gets `S_i,t = F_L(i)(t) F_S(i)(t)`, giving **four pixel states**: `00` static background · `01` **occluded by another object** · `10` static foreground · `11` dynamic foreground. The FSM itself has **five states — MF** (Moving Foreground), **CSF** (Candidate Static Foreground), **OCSF** (Occluded Candidate Static Foreground), **SFO** (Static Foreground Object), **OSFO** (Occluded Static Foreground Object) — and **two timers**: **Count1** = how long a foreground object has been stationary (threshold **T_st**, the time to be called abandoned) and **Count2** = how long a candidate is occluded (threshold **T_sh**, the time for the occluder to be absorbed into the short-term background). Their point: judging on the current pixel state alone, with no temporal context, gives inaccurate results — hence the state sequence.

**SAO-FPN specifics.** Input 640×640×3. The Backbone **removes the C5 module** (final map 40×40×512) before SPPF; the Neck's pyramid is *deepened* to produce **F2 (160×160×128)** and **F1 (320×320×64)**, fused with Backbone C2 and C1; and the PAN's original **P4 (40×40×512) and P5 (20×20×1024) outputs are removed** to cut computation. Final prediction scales: 320×320×64, 160×160×128, 80×80×256 — P1 and P2 retain the finer texture features that small and medium objects need.

**LFEM mechanics.** X1 (lower layer) is **cropped, padded and rearranged** to gather local features, then two linear layers build **K** (feature similarity) and **V** (new output vector); X2 (higher layer) passes through a linear layer to build **Q**. Q·K is point-wise multiplied, normalised, softmaxed into attention weights, used to weight-sum V, and the result is fused back with X2. It is a self-attention transformation that injects low-level detail into the high-level head — which is why occluded objects survive.

**The multi-scale ablation (VisDrone) — genuinely useful design guidance.** Detection branches are named O0…O5 by output feature-map size (O1 outputs 640×640):

| Detection branches | mAP@0.5 | mAP@0.5:0.95 | Params (M) |
|---|---|---|---|
| O3, O4, O5 (default) | 34.5 | 19.2 | 15.8 |
| O2, O3, O4 | 39.4 | 22.2 | 18.7 |
| **O1, O2, O3** ← chosen | **42.0** | **23.5** | 21.6 |
| O0, O1, O2 | 42.1 | 23.6 | 22.8 |

**Reading:** replacing O5 with O2 gives **+4.9 mAP**; replacing O4 with O1 gives another **+2.6**; going all the way to O0 gives **+0.1 for +1.2 M params** — not worth it. *Shift your detection scales down, but stop at O1.*

**Full ablation (Table 5, VisDrone) — note the parameter column, which is the paper's under-sold headline:**

| Structure | mAP@0.5 | mAP@0.5:0.95 | Param (M) | GFLOPS |
|---|---|---|---|---|
| Baseline (YOLOv5s) | 34.5 | 19.2 | 7.0 | 15.8 |
| M1 (O5→O2) | 39.4 | 22.2 | 7.1 | 18.7 |
| M2 (O4→O1) | 42.0 | 23.5 | 7.2 | 21.6 |
| M3 (+ Backbone/Neck simplification) | 42.7 | 23.8 | **1.6** | 15.6 |
| **SAO-YOLO (+ SODHead)** | **43.5** | **24.3** | **1.7** | 18.2 |

**Component contributions:** O5→O2 **+4.9 / +3.0**; O4→O1 **+2.6 / +1.3**; Backbone/Neck simplification **+0.7 / +0.3 *while cutting 5.6 M params and 6.0 GFLOPs***; SODHead **+0.8 / +0.5**. **Overall vs. baseline: +9.0 mAP@0.5 / +5.1 mAP@0.5:0.95 with 4× fewer parameters (7.0 M → 1.7 M).** Their reasoning for the simplification: the original network *"lost some texture information of the detection objects during convolution, connection, and pooling operations"* — reducing depth retains semantic information while cutting detail loss.

**Comprehensive comparison (Table 6, VisDrone, 300 epochs) — the strongest evidence in the paper and one I initially missed:**

| Model | P/% | R/% | mAP@0.5 | mAP@0.5:0.95 | Param (M) |
|---|---|---|---|---|---|
| SSD | 21.1 | 35.6 | 24.1 | 18.8 | 24.5 |
| Faster R-CNN | 43.3 | 35.6 | 33.8 | 21.4 | 41.29 |
| YOLOv3 | 50.3 | 37.4 | 36.0 | 19.4 | 63.07 |
| YOLOv4 | 47.9 | 39.8 | 36.4 | 20.1 | 61.4 |
| YOLOv5s | 46.8 | 34.5 | 34.5 | 19.2 | 7.0 |
| YOLOv6 | 44.6 | 38.5 | 34.8 | 18.5 | 9.67 |
| YOLOv7 | 53.5 | **42.5** | 40.9 | 22.3 | 37.2 |
| YOLOv8s | 53.3 | 40.0 | 41.4 | **25.1** | 11.1 |
| TPH-YOLOv5 | 51.1 | 39.2 | 42.4 | 21.3 | 22.5 |
| **SAO-YOLO (ours)** | **54.2** | 41.4 | **43.5** | 24.3 | **1.7** |

**It beats TPH-YOLOv5 — the strongest small-object YOLO variant — by +1.1 mAP@0.5 and +3.0 mAP@0.5:0.95 with 13× fewer parameters**, and beats YOLOv8s by +2.1 mAP@0.5 with 6.5× fewer. For an edge deployment this is the most compelling accuracy/size trade-off in the corpus.

**Datasets: ABODA · PETS 2006 · AVSS 2007 · VisDrone2019.** (VisDrone = 6 471 / 548 / 1 610, 2.6 M labels, 10 classes, high proportion of small and occluded objects — used to train and ablate SAO-YOLO because the abandonment datasets are far too small.)

**Environment:** NVIDIA GTX 3060, Windows 10, Python 3.9, PyTorch 2.0.0 + CUDA 11.8, **300 epochs, initial LR 0.01**.

**Results:**

*PFSM component evaluation:*

| Method | PETS 2006 R / P | AVSS 2007 R / P |
|---|---|---|
| PFSM-only | 100 / 58.3 | 100 / 43 |
| Improved-PFSM-only | 100 / **63.6** | 100 / **50** |
| **Improved-PFSM + SAO-YOLO** | 100 / **87.5** | 100 / **100** |

The improved PFSM alone gains **+5.3 % (PETS)** and **+7 % (AVSS)** precision, but *"is still prone to false alarms for stationary pedestrians due to its inability to judge the type of stationary objects"* — **the detector is what removes the still-person false alarms.** This is a clean isolation of where each component's value lies.

*ABODA per-video:* **12 TP, 2 FP, R = 100 %, P = 85.7 %.** The 2 FPs are both in **Video 11** (crowded).

*Comparison:*

| Method | ABODA TP / FP / R / P | PETS 2006 R / P | AVSS 2007 R / P | **Combined R / P** |
|---|---|---|---|---|
| **Ours** | 12 / 2 / 100 / 85.7 | 100 / 87.5 | 100 / 100 | **100 / 91.1** |
| Lin et al. 2015 | 12 / 6 / 100 / 66.7 | 100 / 100 | 100 / 100 | 100 / 88.9 |
| Saluky et al. | 9 / 3 / 75 / 75 | 70 / 77 | 72 / 72 | 72.3 / 74.7 |
| Ilya et al. | 9 / 4 / 75 / 69.2 | — | — | — |

**+2.2 % combined precision over Lin et al.**

*Stability analysis — rare and welcome.* Three random-grouping experiments (6 trainings) on VisDrone: **mAP@0.5 range 37.8–38.9 % (spread 1.1 %)**, **mAP@0.5:0.95 range 20.5–21.2 % (spread 0.7 %)**. End-to-end on ABODA across the six models: **recall 100 % every time; precision fluctuates 71–75 %.**

> ⚠️ Note the honest discrepancy: precision is **85.7 %** with the paper's chosen model but **71–75 %** across randomly-retrained ones. The 85.7 % is a best-model number.

**Data Availability Statement — two useful confirmations.** They link ABODA (github.com/kevinlin311tw/ABODA) and VisDrone (aiskyeye.com/download), then state: *"Another dataset, for unknown reasons, is not currently available"* — **independent confirmation that PETS 2006 / AVSS 2007 could not be obtained** even by a 2024 paper that evaluates on them. They also state: *"the authors made datasets suitable for use through their own annotations"* — **independent confirmation of the ABODA annotation-inconsistency caveat in §6.4.**

**Their own stated limitation, which is unusually candid:** *"There is still a scarcity of datasets specifically tailored for abandoned object scenes… As a result, the experimental results presented in this paper might not comprehensively reflect the accuracy of complex real-world scenarios related to abandoned object detection and have certain limitations."* Funding: National Key R&D Program of China, grant 2020YFC0833203.

---
### 3.5 Era 5 — Current state of the art (2025–2026): domain adaptation, depth, foundation models

---

#### **[P25] A System for Real-Time Detection of Abandoned Luggage**
**Ivan Vrsalovic, Jonatan Lerga & Marina Ivasic-Kos** — University of Rijeka / Centre for AI, Croatia · **Sensors 25(9):2872, 2025**

**The single most useful paper in the corpus for practical dataset and detector decisions.** It is the only one that systematically quantifies **domain shift**, **model size vs. object size**, and **dataset size vs. performance** on real CCTV.

**Datasets (all custom but fully documented — see §1.3 for the full table):** **CCTV-Korzo** (60→100 images, outdoor pedestrian zone, Rijeka), **CCTV-Düsseldorf** (240→350 images, indoor Düsseldorf Airport), **CCTV-KD** (combined, **474 images / 9 174 annotations: 6 414 person + 2 760 luggage**), **CCTV-KD-E** (extended, 570 / 11 104).
- Sampling protocol: **one frame per second** from months of 24/7 footage, at varied times of day, chosen because it is *"a sufficient frequency to detect changes in a scene related to the presence of luggage and people's behavior."*
- Annotation policy: only objects **> 10 px** and not heavily occluded were labelled.
- **Object-size composition (COCO criteria): 6 976 small, 2 198 medium, 0 large** — hence **AP_large = −1.0**. This is the defining property of bird's-eye surveillance and why COCO-pretrained models fail.
- **Augmentation** differed per set: Korzo got horizontal flip, saturation ±25 %, exposure ±10 %, noise ≤ 0.1 % of pixels; Düsseldorf got fewer transforms (flip, rotation ±13°, hue ±21°, exposure ±5 %) *because it already contained more scene variety and more luggage*.
- **Test protocol:** all models tested on a **held-out Düsseldorf airport test set none had seen**. Korzo / KD / KD-E split 80/20 (train/val); Düsseldorf split 75/10/15.

**Models compared:** **YOLOv8** (n/s/m/l/x), **YOLOv11** (n/s/m/l/x), **DETR with ResNet-50 backbone**. All COCO-pretrained, then fine-tuned. Training: 2 classes (person, luggage), YOLOv8 default composite loss (BCE classification + distribution focal loss + **CIoU** box loss, IoU-aware classification score), **SiLU**, **batch 16, momentum 0.937, 640×640 input, up to 150 epochs or until loss plateaus**, best checkpoint by validation loss. Tracker: **ByteTrack** (YOLOv11's default), chosen because it associates **low-confidence detections as well as high-confidence ones**, improving identity persistence.

**Result 1 — the domain-shift number (Figure 10), the headline finding:**

| Model | Original (COCO) | Fine-tuned Korzo | Fine-tuned Düsseldorf | Fine-tuned **Korzo+Düsseldorf** |
|---|---|---|---|---|
| YOLOv8-m | **2 %** | 43 | 80 | **86** |
| YOLOv11-m | 2 % | 43 | 80 | **86** |
| DETR (ResNet-50) | 21 % | 28 | 50 | 50 |

(elsewhere stated precisely: **YOLOv8-m mAP@0.5 rises from 3.34 % to 86.44 %**)

**Two conclusions:** (a) COCO-pretrained detectors are **useless** on bird's-eye CCTV without fine-tuning; (b) **combining domains helps** — adding outdoor Korzo images to the indoor airport set gave **+5–6 %** over Düsseldorf alone, because the *perspective* is similar even though the scene isn't. (c) **DETR trails the YOLOs by ~40 points**, attributed to small objects: *"transformers often struggle with detecting small objects due to their global attention mechanism, as they can dilute fine-grained details."* DETR also needs **93.6 ms/frame ≈ 10 FPS** vs 10–24 ms for the YOLOs → unsuitable for real time.

**Result 2 — dataset size (YOLOv8-m):**

| Training set | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| CCTV-KD-65 % | 0.7918 | 0.3729 | 0.8264 | 0.7013 |
| **CCTV-KD** | 0.8644 | **0.5424** | 0.8943 | **0.7768** |
| CCTV-KD-E | **0.8771** | 0.5318 | 0.8916 | 0.7725 |

**Cutting 35 % of the data costs ~7 % across all metrics; adding 20 % more gains ~1 %.** They therefore keep CCTV-KD as the best complexity/performance trade-off. *This suggests a saturation point around ~470 images / ~9 000 annotations for a 2-class CCTV detector.*

**Result 3 — model size (validation, best epoch):**

| Metric | YOLOv8(KD)-n | YOLOv11(KD)-n | -s | -s | -m | -m | -l | -l |
|---|---|---|---|---|---|---|---|---|
| Precision | 0.854 | 0.841 | 0.841 | **0.912** | 0.894 | 0.875 | **0.906** | 0.878 |
| Recall | **0.803** | 0.786 | 0.799 | 0.785 | 0.777 | 0.795 | 0.798 | 0.793 |
| mAP@50 | 0.856 | 0.851 | 0.864 | **0.883** | 0.864 | 0.864 | **0.881** | 0.866 |

Differences are within ±2 % almost everywhere; the exception is **YOLOv11-s beating YOLOv8-s by +7 % precision**. Their selection criterion is explicitly **recall** — *"it is more critical that the model achieves higher recall in order to detect as many potentially suspicious objects as possible."*

**Result 4 — object-size breakdown on the Düsseldorf test set (the decisive table):**

| Metric | v8-s | v8-m | v8-l | v11-s | v11-m | v11-l |
|---|---|---|---|---|---|---|
| mAP@0.5 | 0.864 | 0.864 | 0.881 | **0.883** | 0.864 | 0.866 |
| **AP_small** | 0.844 | 0.833 | 0.845 | **0.858** | 0.828 | 0.844 |
| AP_medium | 0.947 | 0.922 | **0.964** | 0.934 | 0.918 | 0.912 |
| AP_large | x | x | x | x | x | x |

**Chosen model: YOLOv11-s** — best on small objects (85.8 %) *and* fast. **A "small" model won.** (YOLOv8-l is best on medium objects at 96.4 %, but small objects dominate the scene.)

**Abandonment algorithm (Algorithm 1):** frames sampled every **0.25 s**; ByteTrack IDs; **objects < 20 px ignored** (too distant to detect reliably — assumed covered by another camera); for each luggage, check whether a tracked person is within **R_own (suggested 40 px ≈ average person height in their feed)**; if none, start the **abandonment timer (T_threshold = 10 s in their tests)**; also apply a **movement threshold (M_threshold)** comparing coordinates across several consecutive frames to filter camera vibration, and **ignore moving luggage** (assumed towed/conveyed).

**Five scenarios analysed (the most thoughtful behavioural analysis in the corpus):**
1. **Single person abandoning luggage** — works cleanly.
2. **Group with luggage** — ID-based tracking maps each bag to the correct individual(s); if some members leave but others remain, the bag stays supervised. **No false alarms when ownership transfers to a cohesive group.**
3. **Restroom break** — two travellers share one suitcase and take turns leaving. Solved by a **proximity-based timer**: anyone remaining near a bag long enough is **dynamically remapped as its active owner**; and whenever the bag moves (lifted/rolled), ownership transfers to whoever is handling it. Prevents false alarms from consecutive short departures.
4. **Shaking / video disruption** (unstable mounts, aircraft-takeoff vibration) — handled by averaging over a variable number of recent frames plus tailored movement thresholds.
5. **Crowded environments** — **honestly reported as unsolved.** *"Detecting abandoned luggage in densely populated areas is inherently challenging and often impractical, as occlusions and constant motion make it nearly impossible to reliably identify stationary objects."* Failure mechanism: **identity changes of the luggage owner** due to temporary obscuration.

**Their explanation of why older papers report higher accuracy** — worth quoting in a related-work section: *"Earlier research demonstrated higher detection accuracy, but this success can be attributed to the simpler conditions under which their systems were evaluated. These studies predominantly focused on detecting single, isolated, abandoned objects in controlled or less dynamic environments."*

**Deployment note:** small YOLO variants are inherently edge-suitable without knowledge distillation, unlike transformers. In centralised systems, prioritise handling high-res streams over model compression.

**Future work identified:** inter-frame multiscale probabilistic cross-attention; multi-camera setups; better occlusion handling; robust re-identification.

---

#### **[P26] YOLOv11-Based Algorithm for Abandoned Luggage Detection with Dynamic Radius Estimation**
**Vrsalovic, Ivasic-Kos, Pobar & Lerga** — University of Rijeka · **IEEE PICom 2025**, pp. 129–136

The follow-up to P25, and it contributes **two things the field genuinely needed: a depth-based fix for perspective, and a public audit of IITP20's labels.**

**The problem being fixed.** A fixed pixel radius around an object corresponds to **wildly different real-world distances** depending on depth. The standard fix is a **ground-plane homography**, which is what Amin et al. used on IITP20 — but homography *"struggles to accurately measure distances or assess proximity for objects placed on elevated surfaces such as chairs or platforms,"* which IITP20 deliberately contains.

**Methodology:**

1. **Two-stage domain adaptation.**
   - Stage A: fine-tune YOLOv11 on their **CCTV-Surveillance Dataset** (238 video segments → **450 images** after augmentation; **2 676 person + 880 luggage detections**; Korzo outdoor + Düsseldorf airport indoor). SiLU, **batch 16, momentum 0.937, 640×640, ≤ 50 epochs**.
   - Stage B: fine-tune the selected model on **only 100 unique IITP20 images**, ≤ 150 epochs (converged at **90**). *"A small number of additional training images is sufficient"* for adaptation to a new environment.

2. **Dynamic radius estimation via monocular metric depth.** They run **Apple Depth Pro** (Bochkovskii et al., 2025) to recover a depth map. Luggage coordinates are correlated with the depth map, giving a robust distance-from-camera estimate; the **effective interaction radius is then scaled inversely with depth** — closer luggage gets a **larger** pixel radius, farther luggage a **smaller** one. Their published example: **radius 250 px on a nearby object, 125 px on a distant one.** Purpose: *normalise the perceived interaction area across depths.*

3. **Three-state abandonment logic.**
   - **Prerequisite:** the luggage must be **stationary**. Moving luggage is immediately removed from consideration — *"an actively moving object inherently implies a form of current interaction or transport."* Displacement is measured across consecutive frames with a motion threshold filtering sensor noise/jitter; any pending flags are cleared on movement. Notably, they use **presence of movement** only — no direction or velocity analysis.
   - **Unattended:** no associated person within the **dynamically-scaled radius**.
   - **Potential Abandonment:** stationary + unattended for a preset time (**8 s in their figures**), with a visible countdown timer.
   - **Definite Abandonment:** a **second, larger "definite abandonment radius"** (also depth-scaled) is checked for **any** person, not just the owner — accommodating an owner who has stepped just outside the interaction zone, or a bystander passively watching. If nobody is in the wider radius and the potential-abandonment timer expires → **abandoned**.
   - **Inner-proximity timer:** a small inner zone around stationary luggage; a person must remain inside it **longer than a preset duration** before a connection is registered. This filters transient/accidental approaches from genuine, intentional interaction, and — critically — **immediately resets pending abandonment states** when a new connection is established.
   - **Re-mapping instead of long-term tracking:** rather than relying on stable tracker IDs (which degrade in crowds through ID switches), objects are **re-mapped on significant movement changes and at each new detection cycle**.

**Detection results:**

*On the CCTV-Surveillance test set:*

| Metric | YOLOv11-s | YOLOv11-m | YOLOv11-l |
|---|---|---|---|
| **Default (COCO)** Precision | 0.059 | 0.079 | 0.508 |
| Recall | 0.218 | 0.202 | **0.065** |
| mAP@50 | **0.032** | **0.042** | **0.011** |
| **Fine-tuned** Precision | 0.74 | 0.76 | 0.75 |
| Recall | 0.73 | **0.80** | 0.81 |
| mAP@50 | 0.75 | **0.76** | 0.76 |

*On IITP20:*

| Metric | Default | CCTV-only | IITP20-only | **CCTV+IITP20** | Amin et al. (ResNet) |
|---|---|---|---|---|---|
| Precision | 0.52 | 0.76 | 0.94 | **0.96** | not reported |
| Recall | 0.19 | 0.80 | **0.94** | 0.93 | not reported |
| **mAP@50** | 0.29 | 0.76 | **0.93** | **0.93** | **0.7*** (IoU threshold unreported) |

*Every fine-tuned variant — including the one never trained on IITP20 at all — beats the dataset authors' own baseline.* They chose **CCTV+IITP20** after additionally testing on real surveillance images scraped from the Internet, where the combined-domain model generalised best.

**Abandonment results on the *corrected* IITP20 ground truth:**

| Category | Amin et al. (ResNet) | **Proposed (YOLOv11 CCTV+IITP20)** |
|---|---|---|
| **Hard** | 0.74 | **0.90** |
| Medium | 0.85 | 0.85 |
| Easy | **1.00** | 0.95 |
| **Total** | 0.86 | **0.90** |

The gain is concentrated in the **Hard** category (+0.16) — exactly the contextually-relevant cases. They lose a little on Easy, which is where the storage-locker/group-separation relabelling bites.

**The ground-truth audit (its own contribution).** Two systematic label errors found and corrected:
- **Group separation** — one member leaves while the bag stays with others; IITP20 labels this "abandoned". *"Our system does not consider such cases as true abandonment, as the luggage remains implicitly attended by the remaining individuals, even if the primary 'owner' has departed."*
- **Designated storage areas** — luggage placed in **lockers or luggage closets** labelled "abandoned". *"Consistent with real-world security protocols, our system does not classify items placed in such designated areas as abandoned."*

Relabelling went **in both directions**. They note IITP20 also **lacks frame-level ground truth** — videos are only folder-sorted — so the exact moment of abandonment is unrecoverable.

**Stated limitations:** the model and tracker must be fine-tuned per deployment environment; **crowded areas degrade trackers (ID switches)**; the huge variety of luggage shapes/sizes/colours means performance drifts on unseen bag types (mitigable by retraining per site). Future work: better background modelling, refined dynamic radius for complex social interactions and crowds.

---

#### **[P27] Abandoned Bag Detection in Public Areas Using Grounding DINO With Fine-Grained Prompts**
**Tomorn Soontornnapar** — Siam University, Bangkok · **ECTI-CON 2025**

**The only vision-language / foundation-model approach in the corpus — and its most valuable result is a negative one.**

**Methodology:**
1. **Frame sampling:** interval `S = max(⌊N/T⌋, 1)` with target **T = 500** frames, sampled at `Fᵢ = i·S` — evenly spread across the timeline, balancing coverage and cost.
2. **Detection: Grounding DINO** with the text prompt **`"person, bag"`**, returning boxes B, confidences L, labels C. **box_threshold = text_threshold = 0.35.** Normalised centre-format boxes converted to clipped pixel coordinates. Annotated frames + box text files + cropped ROIs saved for downstream analysis.
3. **Loop finding for object association:** centroids `(c_x, c_y) = ((x₁+x₂)/2, (y₁+y₂)/2)`; **Euclidean distance** between person and bag centroids; each person is assigned the **closest not-yet-assigned bag** in the frame; unique IDs maintained for temporal consistency. A **separation frame counter** per person–bag pair; exceeding **T_s** flags the bag as abandoned. Frames are annotated with boxes, connecting lines and "attended"/"abandoned" labels.
4. **Criteria: T_s > 5 frames and C_dist > 170 px** (hop distance 20 in the workflow figure).

**Datasets:** **PETA** (for prompt verification) + **ABODA** (for the actual task).

**Result A — Grounding DINO on PETA with the prompt `"person"` (excellent):**

| Subset | Images | TP | FP | FN | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|
| MIT | 888 | 888 | 0 | 0 | **100 %** | 100 % | 100 % | **100 %** |
| GRID | 1 275 | 1 255 | 0 | 20 | 98.43 % | 98.43 % | 100 % | 99.21 % |
| i-LID | 477 | 475 | 0 | 2 | 99.58 % | 100 % | 99.58 % | 99.79 % |
| CAVIAR4REID | 1 230 | 1 229 | 0 | 1 | 99.92 % | 100 % | 99.92 % | **99.96 %** |

**Result B — the same model with the prompt `"bag"` (poor) — this is the important table:**

| Subset | Images | TP | FP | TN | FN | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| MIT | 888 | 213 | 174 | 364 | 137 | 65.00 % | 61.00 % | 55.00 % | **58.00 %** |
| GRID | 1 275 | 292 | 163 | 348 | 472 | 50.20 % | 38.22 % | 64.18 % | **47.91 %** |
| i-LID | 477 | 135 | 110 | 139 | 93 | 57.44 % | 59.21 % | 55.10 % | **57.08 %** |
| CAVIAR4REID | 1 230 | 116 | 439 | 581 | 94 | 56.67 % | 55.24 % | **20.90 %** | **30.33 %** |

> **The finding to take away:** a state-of-the-art open-vocabulary detector is **essentially solved on `person` (≈100 % F1) and essentially broken on `bag` (30–58 % F1)** at surveillance distances. Diagnosed causes: (1) images captured far from the camera → both person and bag are tiny; (2) **annotation ambiguity** — `carryingOther` visually resembles bags, while `carryingPlasticbags` is not really a bag. This is a direct, quantitative statement of the **small-object + semantic-ambiguity bottleneck** that this whole field runs into.

**Result C — ABODA:**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Park et al. 2019 | **100 %** | **100 %** | **100 %** |
| Newlin/Russel & Selvaraj 2024 | 91.67 % | 100 % | 95.65 % |
| **Proposed (Grounding DINO)** | 83.33 % | 90.91 % | 86.96 % |
| Lin et al. 2015 | 75.00 % | 81.82 % | 78.26 % |
| Dwivedi et al. 2020 | 47.62 % | 90.91 % | 62.50 % |

Correct in most videos with minimal FPs; **fails videos 5 and 6** — both night-vision, where *"the bags appeared either as glare or blended in with the surrounding environment."*

**Note on their metric:** recall is computed against ground-truth instances, and **true negatives / false negatives are not explicitly modelled** because the dataset focuses solely on abandoned-bag detection — so these numbers are not fully commensurate with standard detection metrics.

**Assessment:** a solid demonstration that **zero-shot foundation models are not yet a drop-in replacement** for fine-tuned detectors on this task. Compare directly against Vrsalovic's fine-tuned YOLOv11-s at **AP_small = 85.8 %** on comparable imagery.

---

#### **[P28] No Bag Left Behind**
**Melebari, Alyamani, Alharbi, Albishri, Sindi, Aljedaani & Alafif** — Jamoum University College, Umm Al-Qura University, Saudi Arabia · **ICAISC 2025**

**✅ Code and data public: `https://github.com/ahmadmelebari/No-Bag-Left-Behind.git`** — one of only two papers in the corpus releasing both.

**Their stated gap:** *"Previous studies… have largely overlooked establishing a reliable connection between bags and their owners."*

**Methodology:** deliberately simple and reproducible.
- Data ingestion from a camera → preprocessing → **YOLO (fine-tuned YOLOv8x)** detects bags and persons per frame.
- **Proximity analysis module:** Euclidean distance `d = √((x₁−x₂)² + (y₁−y₂)²)` between bag-box centre and person-box centre, against a **predefined threshold**.
  - No person within threshold ⇒ **unattended**, localised with a **yellow** box.
  - Person within threshold ⇒ **attended**: person **purple**, bag **green**, joined by a **blue line**.
  - **Persistence:** if a bag was flagged unattended in previous frames, it stays yellow; if it's newly unattended, the algorithm **also draws a red box around the person who left it** — the forensic output.
- **GUI (Python `tkinter`)** letting the operator choose the video source (camera or file), **YOLO version**, **device (CPU/GPU)**, **detection confidence**, and **proximity threshold**.

**Dataset:** 🔒→✅ **4 real-world scenarios, publicly released**:
1. Person arrives with a bag, places it, walks away; two others pass by without interacting; owner returns.
2. Three people (bag, backpack, handbag); the bag and backpack owners leave their items; the handbag person passes without interacting; owners return.
3. One person leaves a bag, another leaves a backpack; both return in sequence.
4. One person drops a handbag, walks away, returns.

**Results (person-vs-bag classification, per scenario):**

| Scenario | Accuracy | Precision | Recall | F1 | Confusion |
|---|---|---|---|---|---|
| 1 | 89.69 % | 97.75 % | 91.58 % | **94.57 %** | 40 persons + 51 bags correct; 7 bags→persons, 1 person→bag |
| 2 | 66.77 % | 80.38 % | 79.77 % | 80.08 % | 138 + 169 correct; 54 bags→persons, 46 persons→bags |
| 3 | 71.78 % | 95.39 % | 74.36 % | 83.57 % | 86 + 81 correct; 24 bags→persons, 3 persons→bags |
| 4 | 60.29 % | **100.00 %** | 60.29 % | 75.23 % | 34 + 7 correct; 6 bags→persons, 0 persons→bags |
| **Average** | **72.13 %** | **93.38 %** | **76.50 %** | **83.36 %** |

**Two honest, clearly-illustrated failure modes (each with a figure):**
1. **Small-scale objects, especially a bag lying flat on the floor**, compounded by lighting and camera angle — the model simply fails to recognise it as a critical object.
2. **The nearest-person fallacy** — *"even when the bag is identified as unattended, the algorithm frequently associates it with the nearest person without confirming whether that individual is the actual owner. This can result in scenarios where the model incorrectly assumes the bag is attended by a different person while the owner has actually left, posing potential security risks."*

**Proposed fix (their future work):** **Re-Identification (ReID)** — extract distinguishing features (clothing, colours, body structure) of the person originally associated with the bag, and compare them against whoever is currently near it. *This is the same conclusion Altunay 2018 and Dogariu 2020 reached — the field keeps identifying ReID as the missing piece.*

> ⚠️ Note the internal tension: the abstract claims *"exceptional accuracy"*, but the measured average accuracy is **72.13 %**. The **93.38 % precision / 76.50 % recall** pair is the honest summary.

---

#### **[P29] Unattended Baggage Monitoring in Public Stations**
**Ngo Le The Bach** (British University Vietnam) **& Hamza Mutaher** (Birmingham City University)

**A deliberately low-resource design: no training data at all.**

**Methodology — two stages:**
1. **Background subtraction with ViBe (Visual Background Extractor).** Preprocessing: **histogram equalisation + Gaussian filtering** per frame. ViBe chosen for **simplicity and speed — up to 200 fps** — despite being a decade old, and despite lower accuracy than statistical alternatives. First frame is used as background. Then **object localisation**: **Canny edge detection** for contours → morphological **dilation and erosion** → **size matching** against predefined size parameters → objects static for **> 30 s** have their centroids saved. **Non-maximum suppression** removes redundant boxes.
   - **Illumination adaptation:** each frame is compared against the subtraction model with a **mean-standard-deviation formula**; a high deviation **triggers a reset of the background model**.
2. **Zero-shot classification with CLIP (ViT-B/32)** — no labelled training data required, which is the entire point given ABODA's lack of annotations. Pipeline: load with PIL, convert to RGB, apply CLIP preprocessing (resize, normalise, tensorise), pair image with class-text prompts phrased as natural-language descriptions, encode both, **normalise** (to avoid overfitting), take **cosine similarity**, apply **softmax**, take argmax.
   - **5 classes:** backpack, people, handbag, suitcase, **non-object**.
   - **Decision:** classified as a bag type → **alarm to the security team**; classified as `people` → the background model **updates to absorb it**. (Elegant: misfires are used to correct the background.)
   - They also discuss the FSL spectrum (few-shot K ≤ 5, one-shot K = 1, zero-shot K = 0) and N-way K-shot framing; they settle on zero-shot.

**Dataset: ABODA**, all 11 videos, first frame as background. They state its key limitation plainly: **unannotated**, and labelling it was out of scope.

**Metrics — the Dwivedi et al. (2020) benchmark:** **CODR** (Correct Object Detection Rate), **FAR** (False Alarm Rate), **OSR** (Object Success Rate), derived from TP/FP/FN.

**Results:**

| Video | Scenario | Illumination change | Objects | TP | FP | FN | CODR | FAR | OSR |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Indoor | No | 1 | 1 | 0 | 0 | **100 %** | 0 | 100 % |
| 2 | Outdoor | No | 1 | 1 | 0 | 0 | 100 % | 0 | 100 % |
| 3 | Outdoor | No | 1 | 1 | 0 | 0 | 100 % | 0 | 100 % |
| 4 | Outdoor | No | 1 | 1 | 0 | 0 | 100 % | 0 | 0 % |
| 5 | Outdoor | **Yes** | 1 | 1 | 0 | 1 | 100 % | 0 | 100 % |
| 6 | Outdoor | **Yes** | 2 | 1 | 1 | 1 | **50 %** | **50 %** | 0 % |
| 7 | Indoor | **Yes** | 1 | 0 | 0 | 1 | **0 %** | 0 | 0 % |
| 8 | Indoor | **Yes** | 1 | 0 | 0 | 1 | **0 %** | 0 | 0 % |
| 9 | Indoor | **Yes** | 1 | 1 | 0 | 0 | 100 % | 0 | 100 % |
| 10 | Indoor | **Yes** | 1 | 1 | 0 | 0 | 100 % | 0 | 100 % |
| 11 | Indoor | No | 2 | 0 | 1 | 1 | **0 %** | 0 | 0 % |

**The clean pattern:** **100 % CODR on the 5 sequences with static camera and stable illumination; total failure (0 %) on videos 7, 8 (light switching) and 11 (crowded).** Their stated diagnosis: *"in dynamic background, the project may not accurately localize the object which is likely to mark it as background and ignore it or raise false alarms"*; and at night, *"people and objects will have their shadow and make the background subtraction impossible to detect and extract if the shadow covers the objects."* The few-shot classifier also fails on objects with **small missing edges**.

**Assessment:** a clear demonstration that **zero-shot classification does not rescue a fragile background-subtraction front end** — the failures are all segmentation failures, not classification failures. Also a decent methodological write-up (Saunders' research onion) if you need a methodology-chapter template.

**Future work stated:** apply a tracking algorithm to identify **who dropped the object**, report both together; extend classes to carton box, clutch, duffel, purse.

---

#### **[P30] Detection of Abandoned Objects Based on YOLOv9 and Background Differencing**
**Huajun Song, Jinbo Wang & Yunze Zhang** — China University of Petroleum (East China), Qingdao · **Signal, Image and Video Processing 19:54, 2025**

**Different domain: highway debris** (luggage, goods, spare tires, accident debris improperly secured on vehicles). But the **fusion architecture is directly transferable**, and the ablation is unusually clean.

**Methodology — two complementary detectors whose errors are anti-correlated:**

1. **IDBM (Iterative Difference of Background Modeling).** Dynamic background modelling with **MOG2**, chosen after a comparison against **KNN-based background modelling** and **DPPratiMediod** (BGS library): all three build good backgrounds in simple scenes, *"however, only the MOG2 algorithm achieved better results in videos with complex backgrounds and more vehicles."* The background model is recorded **every 300 frames**, producing a series of vehicle-free background images. Then: **difference the background images built at times k+1 and k** → binarise → **erosion/dilation** → the **outer rectangle of each irregular suspected region** is the object location. *The trick is that it bypasses analysing the abandonment process entirely and just detects when a new object has become part of the background.*
2. **Improved YOLOv9.** YOLOv9-C baseline (PGI + GELAN). Loss modified from **CIoU** to **Inner-CIoU** and **MPDIoU**. Their diagnosis of CIoU: the aspect-ratio term *v* "describes the relative value, which has some ambiguity", and it ignores the easy/hard sample-balance problem, weakening generalisation — and most improved losses just bolt on new terms without addressing the limitations of IoU itself. **Inner-IoU** computes IoU against an **auxiliary bounding box** scaled by a `ratio` factor (`b_l = x_c − w·ratio/2`, etc.), improving generalisation; **MPDIoU** replaces the centroid term with **minimum point distances** between the two boxes' top-left and bottom-right corners, `MPDIoU = IoU − ρ²(P₁ᵖʳᵉᵈ,P₁ᵍᵗ)/(w²+h²) − ρ²(P₂ᵖʳᵉᵈ,P₂ᵍᵗ)/(w²+h²)`, folding overlap, centroid distance and width/height deviation into one metric for faster convergence. They combine the two by applying the Inner-IoU idea to MPDIoU.
3. **Fusion rule:** YOLOv9's detections **guide** the judgment; low-confidence YOLOv9 targets are adjudicated by combining them with the IDBM iterative-difference result. *"Utilizing the difference of the two algorithms on the recognition mechanism."*

**Dataset:** 🔒 **2 813 images** of abandoned objects from **highway surveillance in Jiangsu Province, China** (1280×960 NVR captures at different times, road scenes and angles) + web footage, split **80/20**. Test videos are **human-site simulated highway throwing** at 3 different scenes. Hardware: i5-12400 + GTX 1060 for testing; **2 × Tesla V100 32 GB** for training. Config: **conf 0.25, IoU 0.45, 300 epochs**, defaults elsewhere.

**Results — loss ablation:**

| Method | P | R | mAP@0.5 |
|---|---|---|---|
| YOLOv9 | 0.9108 | **0.9406** | 0.9361 |
| YOLOv9 + MPDIoU | 0.9399 | 0.9174 | 0.9578 (+2.32 %) |
| YOLOv9 + Inner-CIoU | 0.9395 | 0.9119 | 0.9584 (+2.38 %) |
| **Both (ours)** | **0.9425** | 0.9209 | **0.9619 (+2.75 %)** |

**Results — the fusion table, which is the most instructive result in the paper:**

| Method | Scene 1 AP / AR | Scene 2 AP / AR | Scene 3 AP / AR |
|---|---|---|---|
| **IDBM alone** | **8.03 %** / 76.80 % | **10.86 %** / 73.91 % | **13.85 %** / 79.35 % |
| YOLOv9 | 86.67 % / **56.52 %** | 79.42 % / **58.70 %** | 80.28 % / **61.96 %** |
| YOLOv8 | 86.04 % / 53.62 % | 78.12 % / 54.35 % | 80.88 % / 59.78 % |
| RT-DETR | 88.37 % / 55.07 % | 78.79 % / 56.52 % | 79.45 % / 63.04 % |
| **Ours (fused)** | **87.14 % / 88.41 %** | **87.23 % / 89.13 %** | **85.71 % / 91.30 %** |

**The pattern is the whole argument.** Background modelling gives **high recall (~74–79 %) with catastrophic precision (8–14 %)** — it finds everything and flags everything. Deep detectors give **high precision (~79–88 %) with poor recall (~54–63 %)** — *"nearly half of the abandoned objects did not meet the threshold criteria to determine whether they were abandoned objects or ground stains, shadows, or picture noise."* Deep learning's recall failure is specifically **generalisation to untrained object types**: *"the training dataset cannot cover all types of debris."* The fusion recovers **both** — **AP ≈ 86.7 %, AR ≈ 89.6 %**.

**Qualitative:** 24 of 25 objects detected in the frame sequences; the single miss had a **colour close to the road surface** — *"the deep learning method cannot accurately identify this situation, but the probability of this situation is low in practice."* Successfully detects tires, stools, traffic cones, boards; accurate boxes, no overlaps.

**Preprocessing note:** highway **region selection** applied to all three scenes to reduce environmental interference — the same ROI idea as Chen et al.

**Data availability:** 🔒 *"The data utilized in this study are not publicly available due to restrictions imposed by the third-party organization from which they were obtained."* Funding: GF Technology and Innovation Special Zone Program.

---

#### **[P31] A Two-Stage Spatiotemporal CNN-YOLOv9 Framework for High-Precision Real-Time Abandoned Object Detection in Public Surveillance Videos**
**Shah, Khan, Alsuwaylimi & Alenezi** — Iqra National University, Peshawar / Northern Border University, Saudi Arabia · **IEEE Access 14:35983–35997, 2026**

**Architecture — an appearance filter in front of a spatiotemporal verifier.**

*Preprocessing:* frames uniformly sampled at **r_s = 15 FPS** (one every 0.25 s) — `S = {Fᵢ ∈ V | i ≡ 0 (mod ⌊r/r_s⌋)}` — then an automatic **relevance filter R(Fᵢ) ∈ {0,1}** removes frames without useful foreground activity. Normalisation `I_norm = (I − I_min)/(I_max − I_min)`; **cropping to 512×512**; augmentation (Mosaic, Colour Jitter, Random Crop, Flip, MixUp), reported to benefit **ABODA most** because of its greater realism.

*Stage 1 — CNN appearance filter.* Object proposals are classified **suspicious vs. non-suspicious** on spatial features (shape, texture, size, contour complexity), producing `C_CNN(oᵢ)`. Only `C_CNN ≥ τ_app` proceeds. The efficiency argument is formalised: with N objects and a suspicious fraction α, `C_full = N(C_CNN + C_YOLO)` vs `C_filtered = N·C_CNN + αN·C_YOLO`, giving `RR = (1−α)·C_YOLO/(C_CNN + C_YOLO)`.

*Stage 2 — YOLOv9 + Kalman.* YOLOv9 (**PGI** preserving fine detail during training, **GELAN** for backbone efficiency) is used **exclusively for spatial localisation**, transfer-learned on PETS2006 + ABODA. Kalman filtering maintains identity; stationarity is `1ₜ(oᵢ) = 1` if `d(Bₜ, Bₜ₋₁) ≤ ε`, and `T_stationary = Σ 1ₜ`. **Abandonment if `T_stationary ≥ τ = 75 consecutive stationary frames ≈ 5 s at 15 FPS`.**

*Decision fusion:* `C_final(oᵢ) = α·C_YOLO+KF(oᵢ) + (1−α)·C_CNN(oᵢ)`, alarm if `C_final ≥ θ`. Alert payload: YOLOv9 box coordinates, CNN classification, timestamp and frame location.

**Datasets: PETS 2006 + ABODA, merged**, split **70 % train / 15 % validation / 15 % test**, with **1 000 labelled validation images** closely evaluated.
- **Exploratory data analysis** (a nice touch): class distribution is roughly balanced (slightly more non-suspicious); a **bounding-box overlay map** shows **central concentration** — objects cluster in the middle of the FOV, matching public-surveillance geometry (entrances, hallways); the **2-D centroid heatmap** supports positional priors; the **size scatter-histogram** shows most objects are **small**, which they use to justify anchor-box sizing. **Pearson correlation ≈ 0.72 between box width and height** (consistent aspect ratios typical of bags/containers), while **(x, y) correlate minimally with dimensions** — justifying **location-independent anchor design**.

**Training:** Python 3.10.4, PyTorch 1.12.1, CUDA 11.7, **NVIDIA RTX 3090 (24 GB) via Google Colab**, **100 epochs, batch 64**, Binary Cross-Entropy for the classifier; YOLOv9 standard multi-component loss (IoU box regression + objectness + classification) left unmodified.

**Results:**
- **CNN classifier: accuracy 99.35 %, F1 99.05 %** on the held-out 15 % test split.
- **YOLOv9 variant comparison:** **YOLOv9e** best — **precision and recall both > 0.99, F1 ≈ 0.99**; lighter t/s variants notably worse on F1.
- **System: 99.81 % accuracy, 99.67 % precision, 99.01 % recall**, at **30 FPS** real-time inference.

**Stated limitations (to their credit):** sensitivity to extreme occlusion, challenging lighting/weather, and high computational demand limiting some edge deployments; the temporal threshold τ needs per-site adjustment.

> **⚠️ Read this paper's numbers with real caution.**
> 1. **Its PETS 2006 description is factually wrong** — *"recorded in a parking lot in an outdoor location"*. PETS 2006 is an indoor railway station. It also claims PETS 2006 has *"elaborate bounding box annotations, with abandonment event labels such as left luggage, removed objects and vehicle movement"* — PETS 2006 provides XML with trigger times and luggage location, **not per-frame bounding boxes**; Smeureanu & Ionescu had to annotate them by hand.
> 2. **The evaluation is a random 70/15/15 frame-level split of a merged PETS+ABODA pool**, not the standard event-level protocol (does a system raise the right alarm at the right time?). Frames from the same short video almost certainly appear in both train and test, which inflates results.
> 3. **τ = 5 s** vs. the field-standard **30 s** makes alarm timing incomparable.
> 4. **99.81 % accuracy is far out of line** with every carefully-evaluated result in this corpus (Vrsalovic: 90 %; Zhou & Xu: 91.1 % precision; Soontornnapar: 86.96 % F1).
> Cite it as a recent architecture proposal, not as a state-of-the-art benchmark result.

---

#### **[P32] Detection of Abandoned Objects in Video Surveillance Systems: A Comparative Analysis of Rule-Based and AI-Oriented Approaches**
**Vitalii-Oleksandr Pastukh & Anastasiia Deineko** — IT STEP University, Lviv, Ukraine · **Advances in Cyber-Physical Systems 11(1), 2026**

**A review paper**, explicitly self-labelled: it *does not* run original experiments on YOLO26 or RT-DETRv2 and instead analyses published benchmarks, architectural characteristics and deployment constraints. Written with an LLM-assistance declaration (Grammarly).

**Its formalisation of the two families — useful, compact reference material:**

*Rule-based.* Adaptive dual background: long-term **B_L** with very low learning rate (walls, floors) and short-term **B_S** with high rate; static foreground = a logical operation on the two masks. **GMM/MOG2:** `P(Xₜ) = Σ ωᵢ,ₜ · η(Xₜ | μᵢ,ₜ, Σᵢ,ₜ)`, with the **learning rate α as the critical parameter** — too high and a static bag joins the background too fast; too low and lighting changes or foliage cause false positives. **ViBe:** non-parametric — a set of previously observed colour samples per pixel; background if the number of samples within a sphere of radius R exceeds a threshold. Its advantages in AOD are **random updating and spatial propagation** — a background pixel also updates one of its 8 neighbours with probability 1/φ, which **fills ghosting artefacts** left after objects begin moving and enforces spatial coherence.

*AI-oriented, three levels.* **(1) Detection** — YOLOv8-family on CSPDarknet53, recognising semantic class (backpack, bag, suitcase, box) under partial occlusion. **(2) Tracking** — **DeepSORT** = Kalman motion prediction + deep appearance descriptors, maintaining the owner–object link through temporary disappearance. **(3) Spatiotemporal modelling** — **ConvLSTM** layers applying convolutions inside the LSTM transitions, so the model understands that an abandoned object is *the result of a dynamic action*, not merely a static spot.

They also note the tracker does **not** decide abandonment; it supplies stable tracks, and a **logical spatiotemporal rule** (association radius → stationarity → owner outside radius → dwell threshold) flips the state from "owned" to "abandoned" — recorded from three tracked parameters: **track continuity, spatial separation, duration of the stationary unattended state**. **LE-DETR** (RepELAN + TFusion + CMI) is discussed for highway objects, with the Triple Fusion module combining high-level semantics with low-level spatial detail for small objects on complex road textures.

**Their headline comparison table (Table 1):**

| Criterion | Rule-based (GMM, ViBe) | AI-based (YOLOv11/YOLO26, RT-DETR) |
|---|---|---|
| **Accuracy (mAP)** | Low; depends on background clarity and absence of noise | **High (0.75–0.99)** via semantic recognition |
| **Occlusion robustness** | Minimal; occlusion resets the stationary state | High; trackers restore IDs after occlusion |
| **Lighting impact** | **Critical**; shadows and reflections create false objects | Moderate; mitigated by training-data diversity |
| **Computational complexity** | Low; CPU/DSP-friendly | High; needs GPU/NPU for real time |
| **Frame rate** | **Very high (100+ FPS)** | Medium (25–60 FPS) |
| **Setup complexity** | High for the integrator — manual per-scene threshold tuning | Low for the user; high for the developer (data collection/labelling) |
| **False positives** | **High** from dynamic background and artefacts | Low; can distinguish a person from a suitcase |

**Their scenario-selection table (Table 2) — practical deployment guidance:**

| Scenario | Recommended | Rationale |
|---|---|---|
| Indoor, controlled lighting (warehouses, technical corridors, server rooms) | **Rule-based** | Minimal scene dynamics; GMM + temporal threshold gives high FPS with minimal resources |
| Strict budget constraints (legacy facilities, many cameras) | **Rule-based** | Runs on existing DVR/NVR without replacing infrastructure |
| High passenger density (airports, railway stations, subways) | **AI** | Requires tracking the owner through crowds and understanding object/person semantics; DeepSORT/ByteTrack handle the complexity |
| Open spaces, complex weather (plazas, parking lots) | **AI** | Lighting/rain/snow render classical background subtraction inoperable |
| Investigation requirements | **AI** | Yields object type, colour and attributes for response teams, not just a detection |

**Their conclusion:** no universal solution; the **most practical answer is a hybrid** — rule-based methods for rapid static-candidate detection, AI detectors for semantic verification and spatiotemporal behaviour analysis, which *"reduces computational load and the number of false positives while maintaining high accuracy."* On detector choice they favour **YOLO26** over RT-DETRv2 for the accuracy/latency balance (YOLO26x ≈ **57.5 mAP at ~11.8 ms** on an NVIDIA T4 with TensorRT 10), citing the NMS-free end-to-end architecture, MuSGD training optimisation, and superior CPU/edge inference. ⚠️ These YOLO26/RT-DETRv2 figures are **adapted from Ultralytics' own published benchmarks**, not independently measured.

**Value to you:** this is the paper to cite for the **rule-based vs. AI framing** and for **deployment-scenario guidance**. It is not a source of new experimental evidence.

---
## 4. Master comparison table

Sorted by year. **Bold** = public dataset. Metrics are as reported by each paper — see §6.4 before comparing across rows.

| # | Paper (year, venue) | Datasets | Core method | Key result | Speed |
|---|---|---|---|---|---|
| P1 | Thirde, Li & Ferryman (2006, PETS) | **PETS 2006** (created) | Dataset + rule definition | 7 seq × 4 cams, a=2 m/b=3 m/t=30 s | — |
| P2 | Auvinet et al. (2006, PETS) | **PETS 2006** (4 cams) | Median BG + **homographic ground-plane fusion** + spatio-temporal fork heuristic | 6/7 TP, 5 FP; 10–71 cm, +0.0…+2.3 s | 0.42 s/frame |
| P3 | Martínez-del-Rincón et al. (2006, PETS) | **PETS 2006** (cams 1,3,4) | **Double BG** static detection + **Multi-Camera UKF** | Mean 0.164 m, 0.06–0.18 s error (S1,2,3,7) | — |
| P4 | Krahnstoever et al. (2006, PETS) | **PETS 2006** (S1–4,6; 2 cams) | Ellipsoid part-model + integral-image greedy detection; central EKF + Munkres | All events, **1 FA** (sitting man) | **15 fps ×2 streams**, 1-core P4 |
| P5 | Smith, Quelhas & Gatica-Perez (2006, PETS) | **PETS 2006** (cam 3, half-res) | **Trans-dimensional RJMCMC** tracking + bag likelihood + DLT homography | Alarms correct **6/7**; fails S4 (owner confusion) | — (5 runs/seq) |
| P6 | Lv et al. (2006, PETS) | **PETS 2006** (cam 3) | Kalman blob tracker ⊕ **edgelet-boosted human detector** + **Bayesian event inference** | All events **within 9 frames (0.36 s)** | — |
| P7 | Li et al. (2006, PETS) | **PETS 2006** (cam 3, 176×144, 8.3 fps) | **PFR** BG + **context-controlled maintenance** + PCR tracking + layer model + **FSM** | **5/7 (71.4 %), 0 FP**, no scene tuning → **7/7 with 2 scene params**; 7.32 % ID-switch rate | **10 fps**, 2.8 GHz PC (3 months live) |
| P8 | Guler & Farrow (2006, PETS) | **PETS 2006** (**all 28 videos**) | Split-based drop-off tracker ∥ **stationary-object confidence image** + **multi-camera voting** | Cams 3/4 avg **0.8 s**; voting: 0.04–1.04 s, 0.057–0.272 m | real-time modules |
| P9 | Liao, Chang & Chen (2008, AVSS) | **AVSS 2007** + **PETS 2006** | **Foreground-mask sampling** + Cr skin + **improved GHT** + **selective tracking** + MAP | AVSS 3/3; PETS 5 alarms + correct silence on S3 | — |
| P10 | Chang, Liao & Chen (2010, EURASIP) | **AVSS 2007** + **PETS 2006** | (journal extension of P9) | Max err **6.80 s**, mean **2.71 s**; **0 FA** vs 2 and 4 for baselines | **17.37 fps**, D1, 2.66 GHz |
| P11 | Otoom, Gunes & Piccardi (2008, CISP) | **PETS 2007** + web/airport mix (309 imgs) | Shape features (corners/lines/circles/compactness) + BayesNet/C4.5/SMO | **70.4 %** 4-class; **94 %** temporal occl., **90 %** spatial occl. | — |
| P12 | Beleznai et al. (2013, ICCVW) | 🔒 6 stereo indoor seq. | **Trinocular stereo depth (octree)** ∥ dual-BG intensity, IoU fusion + **Motion History** + maximally-stable region growing | **Recall 1.00 everywhere**; precision 0.50–1.00 | **5 fps** (10 fps stereo only), 10×10 m |
| P13 | Ajami & Lang (n.d.) | 🔒 8 tram ONI (RGB-D) | **3 background models × 2 sensors** + histogram arbitration + **SURF** matching + depth-based occlusion counter | 8/8 detected (43–140 s vs 40 s threshold); **0 % FP** (small set) | — |
| P14 | Solus, Ovseník & Turán (2017) | 🔒 4×5 frames | **Optical JTC correlator** + Sobel preprocessing; Δ correlation-peak coordinates | Monotonic Δx 7→67–77 px | optical |
| P15 | **Dahi et al. (2017, CVIU)** | **PETS 2006 · PETS 2007 · AVSS 2007 · CDnet 2014 · ABODA** | **Edge-based** BG + temporal accumulation + hysteresis + edge clustering; **objectness + staticness** scores | **P=R=F1=1.0** on PETS 2006 **and** AVSS 2007, whole scene | **18 fps** @720×576; **108 fps** @320×240 |
| P16 | Santad et al. (2018, GCCE) | 🔒 11 lighting×geometry conditions | YOLO + per-object tracker class + distance threshold + GUI | Success in all 11 conditions (no metrics) | real-time |
| P17 | Altunay et al. (2018, SIU) | **ABODA** (indoor) + **INRIA** + 🔒 3 000 bags | Dual foreground (MoG) → **Faster R-CNN** (region-based, once per object) | ABODA **P 83.33 / R 100**; custom **87.5/87.5** | Faster R-CNN 1.14 s vs YOLO 0.36 s |
| P18 | **Smeureanu & Ionescu (2018, EUSIPCO)** | **AVSS 2007 · PETS 2006 · PETS 2007 · TCD** (14 vids, 42 869 frames) | SOD (fg − motion) + **cascade of 2 GoogLeNets** + **scene-specific synthetic training samples** | Overall **F1 90.29 % frame / 90.18 % pixel** (vs 73.20 baseline) | **~40 fps CPU** |
| P19 | Arora et al. (n.d.) | **PETS 2006** | MoG + texture shadow removal; static regions; CSV feature extraction (colour/date/time) | **5/6 detected**; misses S7 | — |
| P20 | Dogariu et al. (2020, COMM) | **MS-COCO** + **CUHK03** + 🔒 120 CCTV imgs | **Mask R-CNN** shared features for detect + suspect-find + **re-ID** | R50-FPN 3x: 40.2 AP@0.75 / 0.231 s; re-ID **top-1 70.8 %** | 0.231 s/img |
| P21 | **Kim et al. (2022, IEEE CEM)** | **ABODA** + KISA + 🔒 CAU | **HLDNet** = OpenPose keypoints × SSD via **Gaussian confidence reweighting**; D > 2×W for 5 s | **10/11 ABODA**; IoU **0.8789** at dumping; **0 errors** where dual-BG gave 7 FP + 1 miss | 120 k iters ≈ 21 h train (RTX 2080Ti) |
| P22 | Chaitra & Basthikodi (2023, DISCOVER) | ⚠️ **UMN · Avenue** (not abandonment sets) | CNN vs SVM vs RF vs GB; HOG/LBP/colour + transfer learning | GB **0.93/0.89/0.91**; CNN **0.92/0.87/0.89**, AP 0.94 | Tesla V100 |
| P23 | Chen et al. (2024, IET Blockchain) | 🔒 >4 600 airport imgs | **YOLOv5** + **ROI-crossing + frame count**; ray-casting perspective correction; **edge + Hyperledger blockchain** | 5-class avg **93.9 %**; recognition **95.8 %**, retention **96.9 %** | Jetson AGX Xavier ×3 |
| P24 | **Zhou & Xu (2024, Sensors)** | **ABODA · PETS 2006 · AVSS 2007 · VisDrone2019** | **Adaptive dual-BG** + improved **PFSM** (occlusion state) + **SAO-YOLO** (SAO-FPN + SODHead/LFEM) | Combined **R 100 / P 91.1**; ABODA 12 TP/2 FP; **+9.0 mAP@0.5 with 4× fewer params (7.0 M → 1.7 M)**; beats TPH-YOLOv5 by +1.1 mAP with 13× fewer params | GTX 3060, 300 epochs |
| P25 | **Vrsalovic, Lerga & Ivasic-Kos (2025, Sensors)** | ✅ CCTV-Korzo/Düsseldorf/**KD** (474 imgs, 9 174 ann.) — **public, GitHub + Roboflow** | Fine-tuned **YOLOv11-s** + **ByteTrack**; radius + dwell + movement threshold; 5 behavioural scenarios | **mAP@0.5 3.34 % → 86.44 %** after fine-tuning; **AP_small 85.8 %**; DETR 40 pts behind | YOLOv11-s ~10.85 ms; DETR 93.6 ms |
| P26 | **Vrsalovic et al. (2025, PICom)** | **IITP20** (audited) + ✅ CCTV | YOLOv11-m two-stage domain adaptation + **Apple Depth Pro dynamic radius** + inner-proximity timer + re-mapping | IITP20 **mAP@50 0.93** (vs 0.70); abandonment **Hard 0.90** (vs 0.74), **total 0.90** (vs 0.86) | real-time |
| P27 | Soontornnapar (2025, ECTI-CON) | **PETA** + **ABODA** | **Grounding DINO** zero-shot with prompts + centroid distance + frame counter | **`person` F1 ≈ 100 %; `bag` F1 30–58 %**; ABODA **86.96 % F1** | — |
| P28 | Melebari et al. (2025, ICAISC) ✅code | ✅ 4 released scenarios | Fine-tuned **YOLOv8x** + Euclidean proximity + GUI; marks the person who left the bag | Avg **acc 72.13 / P 93.38 / R 76.50 / F1 83.36** | real-time |
| P29 | Ngo & Mutaher (n.d.) | **ABODA** (all 11) | **ViBe** + Canny/morphology + size match + **CLIP ViT-B/32 zero-shot** (5 classes) | **CODR 100 %** on 6 videos; **0 %** on 7, 8, 11 | ViBe up to 200 fps |
| P30 | Song, Wang & Zhang (2025, SIVP) | 🔒 2 813 highway imgs | **IDBM (MOG2 iterative background differencing)** ⊕ **improved YOLOv9** (MPDIoU + Inner-CIoU) | IDBM alone AP 8–14 %; YOLOv9 alone AR 54–62 %; **fused AP 86.7 / AR 89.6** | — |
| P31 | Shah et al. (2026, IEEE Access) | **PETS 2006 + ABODA** (70/15/15 frame split) | CNN appearance pre-filter → **YOLOv9e + Kalman**, τ = 75 frames; weighted fusion | ⚠️ **99.81 % acc / 99.67 % P / 99.01 % R** — see caveats §3.5 | **30 FPS**, RTX 3090 |
| P32 | Pastukh & Deineko (2026, ACPS) | Review (no experiments) | Rule-based vs AI taxonomy; GMM/ViBe formalisation; YOLO26 vs RT-DETRv2 (from vendor benchmarks) | Recommends **hybrid**; scenario-selection table | — |

---

## 5. Methodology taxonomy — the recurring building blocks

Every system in this corpus is assembled from choices in five slots. This is the cheat-sheet.

### 5.1 Slot 1 — Static-object proposal

| Family | Mechanism | Papers | Strengths | Failure modes |
|---|---|---|---|---|
| **Single BG + temporal accumulation** | Threshold vs. one background, accumulate | Auvinet, Arora, Dahi (on edges) | simple, fast | illumination, shadows |
| **Dual / double background** | Long-term (slow α) vs short-term (fast α); their difference = newly static | Martínez-del-Rincón, Porikli-derived: Beleznai, Altunay, Zhou & Xu, Pastukh | resolves drop-off vs removal ambiguity | still vulnerable to illumination + shadows; **7 FP + 1 miss** on outdoor KISA scenes (Kim) |
| **Adaptive dual background** | Learning rate responds to lighting + target complexity | Zhou & Xu 2024 | noise-adaptive | needs the extra impact-factor machinery |
| **Foreground-mask sampling** | **Intersection of 6 masks over 30 s** | Liao 2008, Chang 2010 | appearance-free ⇒ any shape/size/colour/angle | fails in high crowd density |
| **Stationary-object confidence image** | Per-pixel confidence with increment/decrement counters | Guler & Farrow 2006 | **survives full occlusion by moving crowds** | needs colour modelling to separate occluder from object |
| **Edge-based BG + accumulation** | Sobel gradients in X/Y, hysteresis thresholding, `i mod 10` gating | **Dahi 2017** | illumination-robust, **no shadow removal needed**, contours rarely overlap | thin/textureless objects |
| **PFR (Principal Feature Representation)** | Per-pixel table of principal spectral/spatial/temporal features + Bayes rule | Li et al. 2006 | dynamic pixels handled by colour co-occurrence | complex |
| **ViBe** | Non-parametric sample set + random update + spatial propagation | Ngo & Mutaher, Pastukh | **up to 200 fps**, fills ghosting | least accurate; fails on light switching |
| **Depth / disparity BG** | Octree voxel comparison of depth background vs current | **Beleznai 2013**, Ajami & Lang | **immune to shadows and illumination** | only where texture supports stereo matching; highlights create spurious valid disparities |
| **Detector-only (no BG)** | Every frame through a CNN detector | Santad, Dogariu, Melebari, Vrsalovic ×2, Soontornnapar | robust to lighting/PTZ/camera shake | poor recall on untrained object types (Song: 54–62 % AR) |

### 5.2 Slot 2 — Object/person discrimination

| Approach | Papers | Notes |
|---|---|---|
| Size / aspect-ratio / mobility heuristics | Martínez-del-Rincón (H/W ≈ 1 ±5 %), Lv (mobility), Smith (small+slow likelihoods) | zero training cost; confuses ski gear with a standing person (Li et al., PETS S5) |
| Skin colour (Cr in YCbCr) + head-shoulder GHT | Liao, Chang | face is the most visible part under a tilted camera |
| Shape features + classical classifiers | Otoom (corners/lines/circles/compactness → BayesNet/C4.5/SMO) | **70 % on 4 classes** is the realistic ceiling |
| Boosted edgelet part detectors | Lv et al. | generic, trained fully independently of test data |
| Ellipsoid part-model likelihood | Krahnstoever | ~4 memory lookups per hypothesis |
| **CNN classifier** | Altunay (Faster R-CNN 97.4 % vs YOLO 52 %), Smeureanu (GoogLeNet cascade), Shah (CNN pre-filter) | the standard modern answer |
| **CNN detector (single-stage)** | Santad, Melebari, Vrsalovic ×2, Zhou & Xu, Chen, Song, Shah | YOLOv5→v8→v9→v11; small variants often win on small objects |
| **Open-vocabulary / VLM** | Soontornnapar (Grounding DINO), Ngo & Mutaher (CLIP) | **`person` solved; `bag` not** (F1 30–58 %) |
| **Hand-luggage detection** | Kim et al. HLDNet | reframes the problem entirely; class-agnostic via "object in/below the hand" |

### 5.3 Slot 3 — Owner association

| Approach | Papers | Failure mode |
|---|---|---|
| Nearest person at drop-off | Lv, Krahnstoever (r_o = 1 m), Melebari | **the nearest-person fallacy** — Melebari documents it explicitly |
| **Spatio-temporal fork** (object separates from a tracked entity) | Auvinet, Guler & Farrow | fails when the fork is unobserved |
| **Back-tracing / selective tracking** | Liao & Chang (Δt = 60 s), Guler (proposed), Lin 2015 | fails under long occlusion |
| Bag-birth history inspection | Smith et al. | **fails on PETS S4** when a second actor is misread as the owner |
| **Bounding-box intersection** | Dogariu | crude; no metric grounding |
| **Feature-vector re-identification** | Dogariu (Mask R-CNN feats, top-1 70.8 %) | scale-invariant, but 70 % isn't enough alone |
| **Group ownership / transitive attendance** | Vrsalovic ×2 | the current best treatment; **explicitly rejects IITP20's labels** |
| **Proximity timer + dynamic re-mapping** | Vrsalovic PICom | handles restroom-break and hand-over cases |
| ROI-crossing (no owner at all) | Chen et al., Song et al. | only viable where a "correct region" exists |

### 5.4 Slot 4 — Spatial reasoning

| Approach | Papers | Comment |
|---|---|---|
| **Tsai calibration** | PETS 2006 (supplied), Guler (0.08 m cams 3/4; 0.2 m cams 1/2) | Auvinet found the supplied Tsai params gave *worse* orthoimages than a direct homography |
| **Ground-plane homography / DLT** | Auvinet, Martínez-del-Rincón, Smith, Amin (IITP20) | **breaks for objects on chairs/elevated surfaces** |
| **Single-view metrology / vanishing points** | Martínez-del-Rincón (height estimator) | predicts expected person pixel height anywhere |
| **Person-height calibration** | Altunay | pixels→cm from a standard human height — no formal calibration |
| **Owner-width-relative distance (D > 2W)** | Kim et al. | fully calibration-free |
| **Raw pixel thresholds** | Soontornnapar (170 px), Melebari, Vrsalovic Sensors (40 px) | simple; wrong at varying depths |
| **Monocular metric depth (Apple Depth Pro) → dynamic radius** | **Vrsalovic PICom 2025** | the current best answer; 250 px near / 125 px far |
| **Stereo depth** | Beleznai (trinocular, 10×10 m) | most accurate but needs hardware |
| **RGB-D** | Ajami & Lang (Xtion PRO LIVE) | gives free occlusion detection via centroid depth |

### 5.5 Slot 5 — Temporal / event reasoning

| Approach | Papers |
|---|---|
| Simple threshold counter | most papers |
| **Finite State Machine** | Li et al. 2006, Zhou & Xu (PFSM + occlusion state), Evangelio & Sikora, Lin 2015 |
| **MAP / probabilistic confidence with user-tunable ρ** | Liao 2008, Chang 2010 |
| **Bayesian inference over hypotheses + evidence** | Lv et al. 2006 |
| **Kalman / EKF / UKF** | Krahnstoever (EKF), Martínez-del-Rincón (**MCUKF**), Lv (KF), Shah (KF) |
| **Multi-camera voting** | Guler & Farrow (t-second + d-metre correlation) |
| **Weighted score fusion** | Shah (α·C_YOLO+KF + (1−α)·C_CNN) |
| **Three-state machine with two radii** | Vrsalovic PICom (unattended → potential → definite) |
| **Track linking / re-mapping** | Krahnstoever, Vrsalovic PICom |
| **ConvLSTM spatiotemporal modelling** | discussed by Pastukh (not implemented in this corpus) |

---
## 6. Cross-cutting synthesis

### 6.1 The eight failure modes that recur across all 20 years

Ordered by how often they break systems in this corpus.

1. **Crowds.** Universal. Krahnstoever (2006): PETS 2006 *"is only moderately complex compared to a crowded gate in an airport."* Chang (2010): *"even humans cannot notice abandonment reliably."* Kim (2022): *"significantly limited in a crowded situation with more than ten people… a chronic problem in the field."* Vrsalovic (2025): *"inherently challenging and often impractical."* **ABODA video 11 (crowded + small object) is failed by Dahi, Kim, Zhou & Xu (2 FP), Soontornnapar and Ngo & Mutaher.**
2. **Long-duration stationary people.** Krahnstoever's single false alarm is a man reading a magazine on a bench whose arm movements spawned object detections. Dahi's entire **staticness score** exists to reject still persons. Zhou & Xu show the improved PFSM *"is still prone to false alarms for stationary pedestrians due to its inability to judge the type of stationary objects."* Beleznai's **Motion History** cue targets exactly this. Li et al.'s S5 failure is the reverse — ski gear looks like a standing person.
3. **Small objects.** The dominant modern bottleneck. Vrsalovic's CCTV-KD has **6 976 small vs 2 198 medium and zero large** objects. Grounding DINO's `bag` prompt drops to **F1 30–58 %**. Melebari fails on a bag lying flat on the floor. Zhou & Xu's entire architecture (SAO-FPN + SODHead) exists for this. Kim fails ABODA video 11 partly on small-object keypoint failure.
4. **Illumination change, shadows, reflections, night.** Dahi's whole motivation for going edge-based. HLDNet's KISA table shows dual-background producing false objects on illumination change, waves and moving vehicles. Soontornnapar fails ABODA videos 5–6 (night vision: glare or blending). Ngo & Mutaher score **0 % CODR on videos 7–8 (light switching)**. Guler's camera-1 errors trace to shadows and reflections displacing the computed ground position.
5. **Occlusion (short and long).** Short occlusion is largely solved (layer models, hidden dwelling counters, depth-based occlusion tests, track linking). **Long occlusion is not** — Beleznai: *"if a proposed left item is occluded by a dynamic occluder, its importance will decrease and after a while the candidate might disappear."* Liao/Chang lose the owner behind a pillar for ~1.5 s on AVSS Medium/Hard.
6. **Owner misidentification.** Smith fails PETS S4 entirely because the second actor is mistaken for the owner. Melebari documents the nearest-person fallacy with a figure. Auvinet cannot re-identify after blob merge/split. **Three separate papers (Altunay 2018, Dogariu 2020, Melebari 2025) independently conclude ReID is the missing piece.**
7. **Perspective / scale.** A fixed pixel radius means different metres at different depths (Vrsalovic PICom). Homography breaks on elevated surfaces (chairs, platforms) — a case IITP20 deliberately includes. Guler's worst single-camera result is a person walking *toward* the camera. Chen's ray-casting perspective correction addresses the same thing.
8. **Transferred / semi-static objects.** Beleznai's precision drops to 0.5 partly from **a pushed rolling chair and opening/closing doors**. Smith's PETS S2 false positive comes from **trash bins being moved**. Krahnstoever's "ghost" patching exists for this. Song's highway system reads **stopped vehicles** as abandoned when using background modelling alone.

### 6.2 What is actually established (the reliable findings)

1. **Domain shift is catastrophic and cheap to fix.** COCO-pretrained YOLO on bird's-eye CCTV: **mAP@0.5 ≈ 1–4 %**. After fine-tuning on ~470 in-domain images: **86 %**. (Vrsalovic ×2 — measured twice, independently, on two datasets.) *Nothing else in this corpus produces a 20× improvement.*
2. **Dataset size saturates early.** 474 images / 9 174 annotations is near the knee. −35 % data → −7 % across metrics; +20 % data → +1 %. (Vrsalovic Sensors 2025.)
3. **Small models can win.** YOLOv11-**s** beat every larger variant on **AP_small (85.8 %)** and was chosen for deployment. Bigger ≠ better when every object is small. (Vrsalovic Sensors 2025.)
4. **Transformers underperform on this task.** DETR-ResNet50 trailed fine-tuned YOLOs by **~40 mAP points** and ran at **10 FPS vs 42+**, attributed to global attention diluting fine-grained small-object detail. (Vrsalovic 2025; supported by Pastukh's review.)
5. **Purpose-built small-object architectures can be radically smaller, not bigger.** Zhou & Xu's SAO-YOLO reaches **43.5 mAP@0.5 on VisDrone with 1.7 M parameters** — beating YOLOv8s (41.4, 11.1 M), TPH-YOLOv5 (42.4, 22.5 M), YOLOv7 (40.9, 37.2 M) and Faster R-CNN (33.8, 41.3 M). Most of the reduction comes from *removing* depth (pruning C5, P4, P5), which simultaneously **gained** 0.7 mAP. Combined with Vrsalovic's finding that YOLOv11-**s** beat every larger variant on small objects, the message is consistent: **for surveillance-scale objects, shallower and shifted-down beats deeper and bigger.**
6. **Background modelling and deep detection have opposite error profiles, and fusing them works.** Song et al. quantify it cleanly: IDBM alone **AP 8–14 % / AR 74–79 %**; YOLOv9 alone **AP 79–88 % / AR 54–62 %**; fused **AP 86.7 % / AR 89.6 %**. Zhou & Xu show the same shape (PFSM alone P 43–58 %; + SAO-YOLO P 87.5–100 %). Pastukh's review independently recommends the hybrid.
7. **High-level reasoning fixes low-level errors.** Lv et al.'s contextual-rule ablation shows floods of false alarms without drop-off filtering: *"high-level reasoning can eliminate the errors of low-level processing."* Zhou & Xu's PFSM→+detector jump (58.3 % → 87.5 % precision on PETS 2006) is the same effect a decade later.
8. **Multi-camera fusion buys robustness, not accuracy.** Guler's voting didn't improve the good cameras — it **eliminated the bad single-camera results**. Krahnstoever: multiple calibrated views constrain tracking; a bag occluded in one view stays visible in another (his S2 result, where a worker moved garbage bins behind a glass wall).
9. **Synthetic, scene-specific training samples work well.** Superimposing template luggage (and people-with-luggage) onto the estimated background of the target scene bought **~+10 F1** over internet-only training and **~+20 F1** over the single-CNN baseline. (Smeureanu & Ionescu 2018.)
10. **Depth is the cleanest fix for shadows, illumination and occlusion** — Beleznai achieves **100 % recall on all six sequences**; Ajami gets a free, unambiguous occlusion test from centroid depth; Vrsalovic PICom recovers depth **monocularly** with Apple Depth Pro, removing the hardware requirement.
11. **Edges beat intensities for static-region detection.** Dahi's argument (illumination robustness, no shadow removal, contours rarely overlap between moving objects, hysteresis handles partial occlusion) is backed by **P = R = F1 = 1.0** on PETS 2006 *and* AVSS 2007 **over the whole scene**, at 18–108 fps.

### 6.3 Open gaps — where a contribution is available

1. **A large, properly annotated public benchmark.** Every recent paper says it. Amin's IITP20 is the biggest attempt and has **no frame-level ground truth** plus **systematic label errors**. ABODA is **unannotated**. PETS 2006/AVSS 2007 are **effectively unavailable**. Vrsalovic's stated future work is *"generating a set of data from public monitored areas that will include different cases of luggage abandonment."* **This is the field's biggest single gap.**
2. **Standardised evaluation protocol.** Papers variously report: event-level TP/FP, alarm-time error in seconds, frame-level P/R/F1, pixel-level P/R/F1 at IoU 0.2, CODR/FAR/OSR, and plain "accuracy". Detection-area restriction is sometimes silently applied (Lin et al.). Random frame splits are used where event splits are required (Shah). **A shared protocol would be a contribution in itself.**
3. **Owner re-identification integrated into abandonment.** Three papers name it as the missing piece; none implement it end-to-end with quantitative abandonment metrics. Dogariu comes closest (70.8 % top-1 on CUHK03) but reports no abandonment numbers.
4. **Crowded scenes.** Unsolved for 20 years. The specific failure is **owner-identity loss through occlusion**, not object detection. Vrsalovic proposes **inter-frame multiscale probabilistic cross-attention**; Pastukh proposes **ConvLSTM**; neither is implemented here.
5. **Semantic context — what "abandoned" actually means.** Vrsalovic's IITP20 audit is the only serious treatment: **luggage lockers and designated storage areas are not abandonment; group hand-over is not abandonment.** A system that understands *place semantics* (is this a locker? a left-luggage counter? a check-in queue?) would eliminate a large share of real-world false alarms.
6. **Small-object detection at surveillance distance.** Zhou & Xu's multi-scale study (O1,O2,O3) is a good start. Grounding DINO's `bag` collapse quantifies the size of the problem for foundation models.
7. **Night / IR.** Consistently the worst-performing condition (ABODA 5, 7, 8) and barely addressed. Gong et al.'s ZERO-DCE-CB + YOLOv7-BS (cited by Shah, not in this corpus) is one of the very few attempts.
8. **Cross-domain generalisation of the *rule*, not just the detector.** Every system re-tunes τ, radii and thresholds per site. Nobody has proposed a principled way to set them from scene geometry.

### 6.4 ⚠️ Caveats before you cite any number from this corpus

- **ABODA is unannotated.** Every ABODA number is scored against that paper's own labels. Cross-paper ABODA comparisons are indicative at best.
- **Detection-area restriction.** Lin et al. (2015) restricted detection to the platform (AVSS) and waiting zone (PETS) — flagged by Dahi et al. Always check whether a method runs over the whole scene.
- **IoU thresholds vary wildly.** Smeureanu & Ionescu use **IoU > 0.2** for pixel-level correctness; Vrsalovic uses **0.5**; Amin et al. **don't report theirs at all** (flagged by Vrsalovic).
- **Alarm thresholds vary: 5 s (Shah) · 8–10 s (Vrsalovic) · 25 s (Beleznai) · 30 s (PETS standard) · 40 s (Ajami) · 60 s (AVSS).** Alarm-time errors are not comparable across these.
- **Best-model vs. average-model.** Zhou & Xu report **P = 85.7 %** for their chosen model, but their own random-grouping study gives **71–75 %** across retrainings. Smith et al. are the only PETS-era group to report multi-run means.
- **Two papers have significant issues:** Shah et al. (2026) misdescribes PETS 2006 and reports 99.81 % accuracy from a random frame-level split of merged data; Chaitra & Basthikodi (2023) evaluates on crowd-anomaly datasets that contain no abandonment events.
- **Metric transposition:** Chen et al.'s abstract and results section swap the 95.8 % and 96.9 % figures.
- **Vendor benchmarks:** Pastukh's YOLO26/RT-DETRv2 numbers are adapted from Ultralytics' published benchmarks, not independently measured.

### 6.5 If you are building a system today — the evidence-backed recipe

Assembled from what actually replicates across this corpus:

1. **Detector:** fine-tuned **YOLOv11-s or YOLOv8-s** (small variant — it wins on small objects). Not DETR. ~**470 images / ~9 000 annotations** from your own camera geometry is enough; more saturates.
2. **Boost recall cheaply:** add a **background-modelling channel (MOG2 or edge-based accumulation)** in parallel and fuse — this is what recovers the ~30-point recall gap on object types the detector never saw (Song, Zhou & Xu).
3. **Augment with scene-specific synthetics:** superimpose template luggage and people-with-luggage onto your estimated background (**~+10 F1**, Smeureanu & Ionescu).
4. **Tracking:** **ByteTrack** (associates low-confidence detections too) with **re-mapping on movement change** rather than reliance on long-term IDs.
5. **Spatial reasoning:** **monocular metric depth → dynamic radius**, not a fixed pixel radius and not a bare ground-plane homography (which breaks on chairs/platforms).
6. **Ownership:** transitive **group ownership**, an **inner-proximity timer** for intentional interaction, and dynamic owner re-mapping. Plan for **ReID** as the next increment.
7. **Event logic:** three states (**unattended → potential → definite**) with two depth-scaled radii; ignore moving luggage entirely; filter camera vibration with a movement threshold over several frames.
8. **Reject still persons explicitly** — either a **staticness/edge-fragmentation score** (Dahi) or a **Motion History** gradient cue (Beleznai). Do not rely on the detector alone.
9. **Semantics:** exclude designated storage areas / lockers by ROI; consider an **ROI-crossing formulation** if your scene has a well-defined "correct" region (it's far more reliable than owner-distance).
10. **Evaluate at event level with alarm-time error against ground-truth trigger times**, over the whole scene, and report the IoU threshold and the multi-run variance.

---

## 7. Quick index — paper → file

| # | Short name | PDF in `RelatedPaper/` |
|---|---|---|
| P1–P8 | PETS 2006 workshop papers (8) | `Computer_Vision_and_Pattern_RecognitionCVPR06_In_.pdf` *(pp. 47–106; two byte-identical duplicates `(1)` and `(2)` also present)* |
| P5 | Smith, Quelhas & Gatica-Perez | also standalone: `SmithQuelhasGatica-cvpr-pets06.pdf` |
| P9 | Liao, Chang & Chen 2008 | `A_Localized_Approach_to_Abandoned_Luggage_Detection_with_Foreground-Mask_Sampling.pdf` |
| P10 | Chang, Liao & Chen 2010 | `675784.pdf` |
| P11 | Otoom, Gunes & Piccardi 2008 | `Automatic_Classification_of_Abandoned_Objects_for_Surveillance_of_Public_Premises.pdf` |
| P12 | Beleznai et al. 2013 | `Reliable_Left_Luggage_Detection_Using_Stereo_Depth_and_Intensity_Cues.pdf` |
| P13 | Ajami & Lang | `Using_RGB-D_sensors_for_the_detection_of_abandoned_luggage.pdf` |
| P14 | Solus et al. 2017 | `Usage_of_optical_correlator_in_video_surveillance_system_for_abandoned_luggage.pdf` |
| P15 | Dahi et al. 2017 | `1-s2.0-S1077314217300243-main.pdf` |
| P16 | Santad et al. 2018 | `Application_of_YOLO_Deep_Learning_Model_for_Real_Time_Abandoned_Baggage_Detection.pdf` |
| P17 | Altunay et al. 2018 | `Intelligent_surveillance_system_for_abandoned_luggage.pdf` |
| P18 | Smeureanu & Ionescu 2018 | `1803.01160.pdf` **and** `Real-Time_Deep_Learning_Method_for_Abandoned_Luggage_Detection_in_Video.pdf` *(same paper)* |
| P19 | Arora et al. | `Abandoned_object_identification_and_detection_system_for_railways_in_India.pdf` |
| P20 | Dogariu et al. 2020 | `Human-Object_Interaction_...pdf` **and** `Dogariu_COMM_2020.pdf` *(same paper)* |
| P21 | Kim et al. 2022 (HLDNet) | `HLDNet_Abandoned_Object_Detection_Using_Hand_Luggage_Detection_Network.pdf` |
| P22 | Chaitra & Basthikodi 2023 | `Machine_Learning_Approaches_for_Abandoned_Luggage_Detection.pdf` |
| P23 | Chen et al. 2024 | `IET Blockchain - 2024 - Chen - Research on airport baggage anomaly retention detection...pdf` |
| P24 | Zhou & Xu 2024 (SAO-YOLO) | `sensors-24-06572.pdf` |
| P25 | Vrsalovic et al. 2025 (Sensors) | `sensors-25-02872-v2.pdf` *(duplicate `(1)` also present)* |
| P26 | Vrsalovic et al. 2025 (PICom) | `YOLOv11_based_algorithm_for_abandoned_luggage_detection_with_dynamic_radius_estimation.pdf` |
| P27 | Soontornnapar 2025 | `Abandoned_Bag_Detection_in_Public_Areas_Using_Grounding_DINO_With_Fine-Grained_Prompts.pdf` |
| P28 | Melebari et al. 2025 | `No_Bag_Left_Behind.pdf` |
| P29 | Ngo & Mutaher | `Unattended Baggage Monitoring in Public Stations.pdf` |
| P30 | Song et al. 2025 | `s11760-024-03609-z.pdf` |
| P31 | Shah et al. 2026 | `A_Two-Stage_Spatiotemporal_CNN-YOLOv9_Framework_...pdf` |
| P32 | Pastukh & Deineko 2026 | `16pastukh-acps-11-191-96.pdf` |

**Duplicates in the folder (safe to delete):** `Computer_Vision_and_Pattern_RecognitionCVPR06_In_ (1).pdf`, `... (2).pdf`, `Human-Object_Interaction_... (1).pdf`, `sensors-25-02872-v2 (1).pdf`, and one of the Smeureanu & Ionescu pair.

### Key public dataset URLs

- **ABODA** — https://github.com/kevinlin311tw/ABODA
- **No Bag Left Behind** — https://github.com/ahmadmelebari/No-Bag-Left-Behind
- **CDnet 2014** — http://changedetection.net/
- **PETS 2006** — historic: http://www.cvg.reading.ac.uk/PETS2006/data.html (dead; archived snapshot 2017-03-10 on web.archive.org); also cited as ftp://ftp.cs.rdg.ac.uk/pub/PETS2006/
- **PETS 2007** — http://www.cvg.reading.ac.uk/PETS2007/data.html
- **AVSS 2007 (i-LIDS)** — http://www.eecs.qmul.ac.uk/~andrea/avss2007_d.html
- **IITP20** — via Amin, Mondal & Mathew, *IEEE MultiMedia* 28(3):75–87, 2021, doi:10.1109/MMUL.2021.3083701
- **I²R PETS 2006 demos** — http://perception.i2r.a-star.edu.sg/PETS2006/UnAttnObj_C3.htm

---

*Every figure, table value, parameter and quotation above was read directly from the PDFs in `RelatedPaper/`. Where a paper's own numbers are internally inconsistent or its dataset description is factually wrong, that is flagged inline rather than silently corrected.*
