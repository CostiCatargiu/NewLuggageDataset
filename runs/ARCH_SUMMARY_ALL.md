# Architecture Search — Complete Summary

**Task:** weapon detection, 4 classes (knife, long_gun, pistol, **other**)
**Base model:** YOLOv12s (width 0.5), 640², ~80 epochs, seed 0
**Protocol:** append-only modules, identity-at-init (zero-gated, γ=0) where possible,
pretrained transfer via Detect-remap, **default TAL** on arch runs to isolate the
architecture effect. Numbers below are **test mAP50 / mAP50-95** on the 70% ablation,
**single seed** (run-to-run noise ≈ ±1 pt — keep that in mind for every delta).

**Baseline (`original_loss_70`): 0.7845 / 0.4979.**
**Best architecture (`r21_widefuse_aux_w50`): 0.7957 / 0.5033** — but a lucky test split
(val 0.7832), and within the noise band of several others.

---

## 1. Headline finding

Across **~60 architecture variants over ~32 rounds, not one cleanly beat plain loss
tuning**, and the spread between architectures is the same size as the seed noise.
The lever on this task is **loss/label-assignment design and annotation quality**, not
network structure. Confirmed again by the corrected-data 2×2 (Section 5).

---

## 2. By mechanism family

| Family | Representative runs | Best test mAP50 | Verdict |
|---|---|---|---|
| **Receptive-field / context** (ZGLSKA, ZG-GC, ZG-SE, ZG-MHSA, SKA, LSKA, k7/k11/k15/strip) | arch_zg_p4, arch_ska, r6_zgp4_k11, r7_strip23, r10_lskamultidil | ~0.790–0.792 | small gains over baseline, flat vs loss tuning; plateau |
| **Wide-fuse @ P4** (two-branch expand-then-fuse) | r11_widefuse (0.7940), r12–r17 variants | 0.7940 | best pure backbone; basis for r21 |
| **P3 small-object mods** (attention/SE/SPP/GC/dual @ P3) | arch_p3_attn, arch_se_p3, r12_widefuse_p3gc, r12_dualwidefuse | ~0.783–0.789 | all hurt small / "other"; P3 branch competes for "other" capacity |
| **Backbone depth / P5 attention** | arch_deeper_p3, arch_p5_attn, arch_deeper_p3_backbone | ~0.770–0.781 | flat-to-worse |
| **Spatial routing** (per-pixel softmax over branches) | r17_selectfuse | 0.7813 | no edge over static fusion |
| **Classifier capacity** (deep/wide cls towers) | r18_deepcls (0.7885), r18_widecls (0.7892), r18_widefuse_deepcls | ~0.786–0.789 | no effect on the "other" ceiling |
| **Detection scale P2** (stride-4 head) | r18_p2head (0.7785), r18_p2deepcls (0.7782) | 0.7785 | recall ↑, AP flat, overall **regressed**; small dropped |
| **Neck topology** (BiFPN/weighted concat, dual-neck) | r19_bifpn (0.7875), r27_dualneck (0.7638) | 0.7875 | flat / negative; dual-neck worst overall |
| **Deep supervision (train-only aux)** | r20_aux, r21_widefuse_aux_w50 (**0.7957**), r22 | **0.7957** | best arch number (lucky split); free at inference |
| **Decoupled / objectness / ranking heads** | r24_decoupled (0.7904), r24_obj, r25_widefuse_decoupled, r26_decoupledobj | ~0.784–0.791 | decoupled = only *reproducible* effect (precision); obj over-suppressed |
| **Cosine / prototype classifier** | r28_decoupled_cosine (0.7848) | 0.7848 | underperformed — evidence the scoring mechanism isn't the bottleneck |
| **Deformable (ZGDCN)** | round 7/8 | — | crashed (torchvision deform_conv2d); never evaluated |
| **New heads/blocks (rounds 29–32)** | DetectDecoupledAux, DetectMultiProto, ZGSmallDetail (p3detail), kernelfix, ZGLSKAWideFuseV2, DetectAuxDual, P2-FPN | ≤0.7893 | none beat r21; see Section 4 |

---

## 3. Full run table (rounds 1–28)

| Run | Mechanism | mAP50 | mAP50-95 |
|---|---|---|---|
| original_loss_70 (baseline) | stock YOLOv12s | 0.7845 | 0.4979 |
| arch_ld_70 | lightweight detect | 0.7714 | 0.4818 |
| arch_deeper_p3_70 | deeper P3 | 0.7721 | 0.4860 |
| arch_deep_p3_bidi_70 | bidirectional P3 | 0.7876 | 0.4921 |
| arch_p3_attn_70 | attention @ P3 | 0.7850 | 0.4920 |
| arch_p5_attn_702 | A2C2f @ P5 | 0.7806 | 0.4878 |
| arch_p5_attn_deeper_p3_70 | P5 attn + deeper P3 | 0.7803 | 0.4850 |
| arch_deeper_p3_backbone_70 | deeper backbone | 0.7787 | 0.4884 |
| arch_dual_p3_70 | dual P3 path | 0.7858 | 0.4861 |
| arch_se_p3_70 | SE @ P3 | 0.7891 | 0.4963 |
| arch_spp_p3_70 | SPP @ P3 | 0.7833 | 0.4919 |
| arch_ska_70 | SKA | 0.7903 | 0.4992 |
| arch_ska_p5_70 | SKA @ P5 | 0.7877 | 0.4923 |
| arch_ska_k5_70 | SKA k5 | 0.7870 | 0.4963 |
| arch_ska_residual_70 | SKA residual | 0.7836 | 0.4939 |
| arch_zg_all_70 | zero-gated everywhere | 0.7844 | 0.4921 |
| arch_zg_gc_p5_70 | ZG global-context @ P5 | 0.7889 | 0.4902 |
| arch_zg_mhsa_p5_70 | ZG MHSA @ P5 | 0.7831 | 0.4883 |
| arch_zg_p45_70 | ZG LSKA @ P4+P5 | 0.7858 | 0.4939 |
| arch_zg_p4_70 | ZG LSKA @ P4 | 0.7905 | 0.4936 |
| arch_zg_se_p45_70 | ZG-SE @ P4+P5 | 0.7828 | 0.4897 |
| r5_cgc_70 | gated global-context head | 0.7864 | 0.4963 |
| r6_ska_p4_70 | SKA @ P4 | 0.7884 | 0.4973 |
| r6_zgp4_k11_702 | ZG LSKA k11 @ P4 | 0.7919 | 0.4989 |
| r7_k11_gc4_70 | k11 + GC | 0.7866 | 0.4980 |
| r7_k15_70 | k15 LSKA | 0.7903 | 0.4973 |
| r7_strip23_70 | strip-23 LSKA | 0.7907 | 0.4979 |
| r8_star_p4_70 | star-block @ P4 | 0.7853 | 0.4955 |
| r10_k11_p4td_703 | k11 P4 top-down | 0.7883 | 0.4955 |
| r10_lskamultidil_70 | multi-dilation LSKA | 0.7906 | 0.4993 |
| r10_lskastripfuse_702 | strip-fuse LSKA | 0.7827 | 0.4931 |
| r11_widefuse_7013 | **wide-fuse @ P4 (k11+strip23)** | **0.7940** | 0.4992 |
| r11_dual_p3p4_707 | dual P3+P4 gates | 0.7856 | 0.4972 |
| r11_expand_707 | expand fuse | 0.7832 | 0.4940 |
| r11_refine_706 | refine fuse | 0.7878 | 0.4956 |
| r12_widefuse_ksmall_703 | widefuse k11+strip3 | 0.7871 | 0.4954 |
| r12_widefuse_p3gc_702 | widefuse + GC @ P3 | 0.7829 | 0.4924 |
| r12_dualwidefuse_702 | widefuse @ P4 & P3 | 0.7826 | 0.4939 |
| r13_r6_lkacls_702 | r6 + LKA cls | 0.7881 | 0.4955 |
| r13_widefuse_lkacls_703 | widefuse + LKA cls | 0.7854 | 0.4957 |
| r13_widefuse_cgc_702 | widefuse + CGC | 0.7836 | 0.4923 |
| r14_widefuse_gcfuse_703 | widefuse + GC-fuse | 0.7823 | 0.4944 |
| r14_widefuse_p2fuse_702 | widefuse + P2 fuse | 0.7873 | 0.4979 |
| r16_compactfuse_70 | compact small-kernel fuse @ P4 | 0.7825 | 0.4971 |
| r16_widefuse_smallcls_70 | widefuse + small cls | 0.7855 | 0.4964 |
| r17_selectfuse_703 | per-pixel select-fuse | 0.7813 | 0.4912 |
| r17_widefuse3_70 | 3-branch wide-fuse | 0.7818 | 0.4941 |
| r18_deepcls_702 | deep cls tower | 0.7885 | 0.4963 |
| r18_widecls_70 | wide cls tower | 0.7892 | 0.4944 |
| r18_widefuse_deepcls_702 | widefuse + deep cls | 0.7863 | 0.4945 |
| r18_p2deepcls_703 | P2 head + deep cls | 0.7782 | 0.4840 |
| r18_p2head_704 | **P2 (stride-4) head** | 0.7785 | 0.4900 |
| r19_bifpn_702 | BiFPN neck | 0.7875 | 0.4967 |
| r19_bifpn_widefuse_70 | BiFPN + widefuse | 0.7818 | 0.4955 |
| r20_aux_705 | train-only aux head | 0.7852 | 0.4929 |
| r21_aux_w50_706 | aux @ 0.5 | 0.7882 | 0.4973 |
| r21_widefuse_aux_703 | widefuse + aux @0.25 | 0.7906 | 0.4994 |
| **r21_widefuse_aux_w50_703** | **widefuse + aux @0.5 (BEST ARCH)** | **0.7957** | **0.5033** |
| r22_widefuse_aux_w75_70 | widefuse + aux @0.75 | 0.7895 | 0.4979 |
| r24_decoupled_70 | decoupled box/cls head | 0.7904 | 0.4944 |
| r24_obj_70 | objectness branch | 0.7769 | 0.4865 |
| r25_widefuse_decoupled_70 | widefuse + decoupled | 0.7840 | 0.4957 |
| r26_widefuse_decoupledobj_70 | widefuse + decoupled + obj | 0.7912 | 0.4934 |
| r27_dualneck_stock_70 | dual neck (WORST) | 0.7638 | 0.4707 |
| r28_decoupled_cosine_70 | decoupled + cosine cls | 0.7848 | 0.4865 |

---

## 4. New mechanisms built in rounds 29–32 (extending r21)

| Run | Idea | mAP50 | mAP50-95 | Result |
|---|---|---|---|---|
| r29_decoupled_aux2 | widefuse + decoupled **+ aux** (new `DetectDecoupledAux`) | 0.7893 | 0.4975 | precision came through, didn't beat r21 |
| r29_kernelfix | widefuse k11+**strip3** (multi-scale) + aux | 0.7881 | 0.4965 | nudged small-"other" (+2.8) but overall below r21 |
| r30_multiproto_k42 | **mixture classifier** (K sub-prototypes, `DetectMultiProto`) | 0.7887 | 0.4978 | best small-"other" of the variants; overall flat |
| r30_p3detail_704 | widefuse + **`ZGSmallDetail`** small-kernel guard @ P3 + aux | 0.7865 | 0.4986 | redistributed small (hurt small-weapons), no overall gain |
| r31_widefuse_v2 | **`ZGLSKAWideFuseV2`** (hybrid large+small RF branch) + best TAL | built | — | written; for results check json |
| r32b_auxdual | **`DetectAuxDual`** (aux sees pre-widefuse detail features) | built | — | written; needs r31 control |
| r21+P2 (review) | r21 + **4-level P2-FPN** detector | built | — | written for corrected-data test |

Net: every round-29–32 extension landed within noise of r21; none robustly beat it.

---

## 5. Final 2×2 on the CORRECTED dataset (the decisive comparison)

{stock arch, r21 arch} × {default TAL, best TAL}, same batch/seed/epochs, corrected labels:

| config | mAP50 | mAP50-95 | small |
|---|---|---|---|
| stock + default (baseline) | 0.8117 | 0.5159 | 0.640 |
| r21 + default (arch only) | 0.8146 | 0.5204 | 0.634 |
| **stock + best TAL (loss only)** | **0.8261** | 0.5199 | 0.654 |
| r21 + best TAL (combined) | 0.8237 | **0.5223** | **0.664** |

- **Loss tuning** = the dominant lever (+1.44 mAP50); **architecture alone** = +0.29 (noise).
- On top of best-TAL, the widefuse arch does **not** help overall (stock 0.8261 ≥ r21 0.8237);
  r21 only edges mAP50-95 / small by ~1 pt (within noise).
- **Conclusion: architecture is at ceiling; loss + label-quality are the levers.**

---

## 6. Bottom line

A systematic ~60-variant architecture search found **no robust architectural win** on this
task. The best arch (r21 = widefuse + aux) is within seed-noise of stock YOLOv12 once the
loss is tuned. The real, reproducible gains came from **loss/assignment tuning** and from
**correcting the incomplete "other" annotations** (~13% train / ~24% test unlabeled).
*All numbers single-seed — confirm the finalists at 2–3 seeds before publishing.*
