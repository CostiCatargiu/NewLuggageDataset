# Diagnostics — model-agnostic, shared by both detectors

None of these train. They analyse runs you point them at, which is why they sit
outside MODEL_v12/ and MODEL_v26/ rather than being duplicated into both.

    collect_confusion_v6i.py   walks a runs tree, gives each run its own folder with
                               confusion matrices + results.csv, and re-runs val to
                               dump the matrix AS NUMBERS (the PNG is a render).
                               -> confusion_collected/

    diag_miss_vs_score.py      re-scores at conf=0.001 and splits every "missed" GT
                               into PROPOSAL failure (no box at any confidence) vs
                               SCORING failure (correct box exists, scored too low).
                               -> miss_Scoreresults/

    diag_threshold_sweep.py    per-class P/R/F1 across confidence thresholds.
                               Tunes on val, reports on test — fitting and reporting
                               a free parameter on the same split is leakage.
                               -> thr_sweep/

    diag_anchor_footprint.py   positives-per-GT vs object area.

## What they established (YOLO26, but the method applies to both)

1. Misclassification is a CONSTANT: 3.94-4.90% across all 81 runs. Twelve loss
   mechanisms, three architectures, four assignment schemes — nothing moved it.
   Everything is a recall problem.

2. The misses are SCORING, not PROPOSAL. true_miss is only 2.7-5.5% per class;
   the model is ~95% capable and is being read at ~76%.

3. But the ceiling is UNREACHABLE by thresholding. Recovering bag's 26.8% of
   low-scoring true positives drags in 20,786 false positives — precision 5.8%.
   The correct boxes exist and are ranked BELOW junk.

   -> The binding constraint is CONFIDENCE RANKING. Every mechanism tried operated
      on assignment, localisation or box regression. None touched score ordering.
      That explains the flat campaign, and it is measured rather than argued.

4. Free win: per-class thresholds, tuned on val and applied to test (transfer is
   near-exact). The optimum is HIGHER than 0.25, not lower.
       micro @ 0.25    P 73.4  R 80.0  F1 76.6
       micro @ tuned   P 83.4  R 74.0  F1 78.5    +1.9
