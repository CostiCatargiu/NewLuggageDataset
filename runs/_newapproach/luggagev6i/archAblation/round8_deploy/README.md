# Round 8 deploy — SNL1 + SCB on YOLO26

Two loss mechanisms designed **for** YOLO26 rather than ported into it. Copy this
folder to the training machine and follow the four steps below in order.

```
round8_deploy/
  patch/utils/loss.py            -> <ultralytics>/utils/loss.py
  patch/utils/tal.py             -> <ultralytics>/utils/tal.py
  patch/cfg/default.yaml         -> <ultralytics>/cfg/default.yaml
  cfg_yaml/y26_p2_dysample.yaml     the architecture (only for the smoke test)
  verify_patch_v6i.py               installs + verifies the patch
  run_yolo26_round8_v6i.py          the 4 runs
```

---

## What these mechanisms are

**SNL1 — Scale-Normalised L1.** YOLO26 is DFL-free (`reg_max: 1`) and replaces
DFL with an L1 on ltrb normalised by **image** size, so the target magnitude — and
the gradient — is proportional to object size. At 640 px:

```
GT side    8 px -> target 0.0063     1x gradient
GT side  256 px -> target 0.2000    32x gradient
```

For the *same relative* localisation error a 256 px trolley contributes ~32x the
regression gradient of an 8 px bag. YOLOv12 does not have this problem: DFL is a
cross-entropy over bins whose magnitude is independent of the target value. The
bias is specific to the DFL-free head. SNL1 divides the residual by the GT's own
extent^p, renormalised to mean 1 so `p` redistributes gradient without rescaling
the term. `p = 0` is bit-identical to stock.

**SCB — Size-Conditioned Beta.** `align_metric = score^alpha * IoU^beta` selects
positives. In the one2one branch `topk2 = 1`, so this picks **the single anchor**
per GT in the branch that carries ~90% of the loss by the last epoch and produces
every prediction (the head is NMS-free). IoU is a high-variance ranking signal on
small boxes and a stable one on large boxes, so one global beta over-trusts IoU
exactly where it is least reliable. SCB interpolates beta by GT size.

---

## Bound your expectations

```
term          gain   share   scale behaviour
box  (CIoU)    7.5   78.9%   IoU is a ratio -> already scale-INVARIANT
cls  (BCE)     0.5    5.3%   size-independent
dfl  (L1)      1.5   15.8%   the biased term SNL1 corrects
```

The defect is real but lives in ~16% of the loss, and the dominant regression term
is already scale-fair. Do not expect anything like YOLOv12's +0.86.

For contrast, the **ported** mechanisms on this same architecture (round 7) all
came out below the control band:

```
SWA alone                   -0.48  (-2.5 sd)
LB-TAL uniform              -0.43  (-2.2 sd)
LB-TAL {4:2,8:3,16:4,32:4}  -0.82  (-4.3 sd)
SWA + LB-TAL                -0.82  (-4.3 sd)
```

---

## Control

Ten runs of this exact architecture at b32/640/seed 0 (rounds 4-6, whose budget
labels never took effect) form a replicate distribution:

```
56.08 +- 0.19   overall mAP50-95   (n = 10)
52.14 +- 0.23   small
66.38 +- 0.49   medium
57.21 +- 2.11   large      <- do NOT read large at n=1

DECISION BAND   55.70 .. 56.46    inside = null, outside = real
```

56.46 is also the best of the ten draws, which is the correct bar: to claim a gain
a config must beat the luckiest replicate of doing nothing.

Note `sd 0.19` is a **lower bound** — those ten share seed 0 and differ only
through nondeterminism. True seed-to-seed spread is likely wider.

---

## Step 1 — install and verify

```bash
python verify_patch_v6i.py --ref /path/to/round8_deploy/patch --install --runtime
```

`--ref` must point at a directory containing `utils/` and `cfg/`, which is what
`patch/` is. Originals are kept as `.bak`.

The line that matters:

```
loss.py READS use_lbtal in v8DetectionLoss   : True
```

**Re-run this after any `pip install -e .`** — a reinstall silently reverts every
patched file. That is how rounds 4-6 produced ten identically-configured runs
under ten different names.

---

## Step 2 — smoke test (10 minutes, do not skip)

This code has never been executed. It was verified by AST parsing and pure-Python
arithmetic only.

```bash
CFG=cfg_yaml/y26_p2_dysample.yaml
DATA=/home/constantin/Doctorat/LuggageDataset.v6i.yolov12/data.yaml

yolo train model=$CFG data=$DATA epochs=2 imgsz=640 batch=32 seed=0 name=smoke_p0
yolo train model=$CFG data=$DATA epochs=2 imgsz=640 batch=32 seed=0 l1_scale_p=0.5     name=smoke_p50
yolo train model=$CFG data=$DATA epochs=2 imgsz=640 batch=32 seed=0 tal_beta_small=3.0 name=smoke_scb
```

Pass conditions:

- `smoke_p0` completes — the new code paths and the `LOGGER` import are sound
- `smoke_p50`'s **l1_loss** column differs from `smoke_p0`'s — SNL1 actually fires
- `smoke_scb`'s losses differ from `smoke_p0`'s — SCB reaches the assigner

If p50 or scb match p0 exactly, the mechanism is a silent no-op. Stop there.

---

## Step 3 — the runs

```bash
python run_yolo26_round8_v6i.py
```

Four runs, ~6.6 GPU-h, architecture frozen at the DySample P2 variant:

```
1  y26_dfl3        dfl 1.5 -> 3.0          PROBE, no code change
2  y26_snl1_p25    l1_scale_p = 0.25       partial bias removal
3  y26_snl1_p50    l1_scale_p = 0.50       half-way to scale-invariant
4  y26_scb_b3      tal_beta_small = 3.0    size-conditioned beta
```

**`y26_dfl3` runs first by list order.** It doubles the L1 term's weight with no
code change. If it lands inside 55.70..56.46 the model is not limited by that term,
correcting its internal scaling cannot help either, and you can kill the remaining
three and save 5 hours.

The runner asserts at epoch 1 that each mechanism is **live in the constructed
criterion**, not merely requested:

```
[guard] SNL1 live on one2one: p=0.25
[guard] SCB live on one2one: beta 3.0 -> 6.0 @ 64.0px
[guard] dfl gain = 1.5
```

A missing guard line means the run is not measuring what its name says.

To run a single config:

```bash
python run_yolo26_round8_v6i.py y26_dfl3
```

---

## Step 4 — read the results

Overall mAP has sd 0.19 across the control, so it resolves ~0.4 pp differences.
**Small** is where SNL1 should show up first if it does anything — run the per-size
eval, do not judge from overall alone:

```bash
python CocoEvalAllFolders_luggage.py   # on each best.pt
```

`p = 0` (control), `p = 0.25`, `p = 0.50` give three points. A monotone trend in
small-object AP is evidence even if overall stays inside the band.

**Do not read the large column at n=1** — control sd there is 2.11 pp, and every
large-object "finding" in rounds 4-6 turned out to be that noise.

SNL1 pushes gradient toward small objects, which on this dataset already costs
large-object AP. If large falls off a cliff between p=0.25 and p=0.50, the useful
range is below 0.25 and p=1.0 is pointless.
