#!/usr/bin/env python3
"""
Generates the framework block diagram for the unattended-luggage paper.

Every text run is measured with PIL (Liberation Sans == Arial metrics) and
emitted at an absolute x, with text-anchor="start". No <tspan dy> and no
text-anchor="middle" anywhere, because several SVG rasterisers (cairosvg,
some Office importers) mis-compute the advance width of centred text that
contains tspans, which silently garbles every equation in the figure.

Mini-markup accepted by every text call:
    *x*        italic
    **x**      bold
    _{abc}     subscript   (may contain *italic*)
    ^{abc}     superscript (may contain *italic*)

Outputs fig2_framework_pipeline.svg and .png next to this script.

    python make_fig2_pipeline.py
"""

import os
import re
from xml.sax.saxutils import escape

from PIL import ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_SVG = os.path.join(HERE, "fig2_framework_pipeline.svg")
OUT_PNG = os.path.join(HERE, "fig2_framework_pipeline.png")

W, H = 1500, 770
PNG_SCALE = 3.0

FONT_STACK = "Liberation Sans, Arial, Helvetica, sans-serif"
FDIR = "/usr/share/fonts/truetype/liberation"
FONT_FILES = {
    ("n", "n"): f"{FDIR}/LiberationSans-Regular.ttf",
    ("n", "i"): f"{FDIR}/LiberationSans-Italic.ttf",
    ("b", "n"): f"{FDIR}/LiberationSans-Bold.ttf",
    ("b", "i"): f"{FDIR}/LiberationSans-BoldItalic.ttf",
}
REF = 400  # measure once at a big size, then scale — keeps fractional sizes exact
_FONTS = {k: ImageFont.truetype(v, REF) for k, v in FONT_FILES.items()}

SUB_SCALE, SUB_DY = 0.72, 0.30
SUP_SCALE, SUP_DY = 0.70, -0.42

# ----------------------------------------------------------------- palette
C_TXT = "#212529"
C_MUTE = "#495057"
C_HEAD = "#6C757D"
NEW_F, NEW_S = "#FDF3E0", "#C07A16"     # proposed contribution
NEW_D = "#8A5A0B"                        # proposed, text
STD_F, STD_S = "#E8EEF6", "#3C6E9F"     # off-the-shelf component
NEU_F, NEU_S = "#F2F3F5", "#6C757D"     # neutral / input
ALM_F, ALM_S, ALM_T = "#FBE3DF", "#B03A2E", "#7B2318"
OK_F, OK_S, OK_T = "#E6F2E8", "#3E7D52", "#20502F"
ARROW = "#343A40"

svg = []
warn = []


# ------------------------------------------------------------- text engine
_TOKEN = re.compile(r"(\*\*|\*|_\{|\^\{|\})")


def _runs(markup):
    """markup -> [(text, weight, style, 'base'|'sub'|'sup'), ...]"""
    out, weight, style, level = [], "n", "n", "base"
    for tok in _TOKEN.split(markup):
        if tok == "":
            continue
        if tok == "**":
            weight = "b" if weight == "n" else "n"
        elif tok == "*":
            style = "i" if style == "n" else "n"
        elif tok == "_{":
            level = "sub"
        elif tok == "^{":
            level = "sup"
        elif tok == "}":
            level, style = "base", "n"
        else:
            out.append((tok, weight, style, level))
    return out


def _width(markup, size):
    w = 0.0
    for text, weight, style, level in _runs(markup):
        s = size * (SUB_SCALE if level == "sub" else SUP_SCALE if level == "sup" else 1.0)
        w += _FONTS[(weight, style)].getlength(text) * s / REF
    return w


def text(x, y, markup, size=12, weight="n", fill=C_TXT, anchor="middle",
         box=None, label=""):
    """Emit one logical line. anchor: 'middle' | 'start' | 'end' (x is that edge)."""
    runs = _runs(markup)
    if weight == "b":
        runs = [(t, "b", st, lv) for t, _w, st, lv in runs]
    total = 0.0
    for t, wgt, st, lv in runs:
        s = size * (SUB_SCALE if lv == "sub" else SUP_SCALE if lv == "sup" else 1.0)
        total += _FONTS[(wgt, st)].getlength(t) * s / REF

    if box is not None and total > box - 8:
        warn.append(f"  OVERFLOW {total:6.1f}px in {box}px box :: {label or markup[:48]}")

    cx = x - total / 2 if anchor == "middle" else x - total if anchor == "end" else x
    for t, wgt, st, lv in runs:
        s = size * (SUB_SCALE if lv == "sub" else SUP_SCALE if lv == "sup" else 1.0)
        dy = size * (SUB_DY if lv == "sub" else SUP_DY if lv == "sup" else 0.0)
        adv = _FONTS[(wgt, st)].getlength(t) * s / REF
        attrs = [f'x="{cx:.2f}"', f'y="{y + dy:.2f}"', f'font-size="{s:g}"']
        if wgt == "b":
            attrs.append('font-weight="bold"')
        if st == "i":
            attrs.append('font-style="italic"')
        if fill != C_TXT:
            attrs.append(f'fill="{fill}"')
        svg.append(f'<text {" ".join(attrs)}>{escape(t)}</text>')
        cx += adv
    return total


def block(x, y, w, h, fill, stroke, sw=1.6, rx=8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
               f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def arrow(d, stroke=ARROW, sw=2, dash=None, marker="ah"):
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}"'
               f'{ds} marker-end="url(#{marker})"/>')



# =============================================================================
# LAYOUT GRID — every number below derives from these constants, so the columns
# share exact edges instead of being eyeballed.
# =============================================================================
M = 28                                   # page margin, left and right
GAP = 36                                 # horizontal gap between stage columns
COL_W = [150, 320, 172, 220, 262, 140]   # sum 1264 ; 1264 + 5*36 + 2*28 = 1500
COL_X = []
_x = M
for _w in COL_W:
    COL_X.append(_x)
    _x += _w + GAP
CX = [x + w / 2 for x, w in zip(COL_X, COL_W)]   # column centres
RIGHT = COL_X[-1] + COL_W[-1]                    # 1472

BAND_T, BAND_B = 70, 340                 # every stage box lives inside this band
SPLIT_GAP = 40                           # gap between the upper and lower rows
ROW_H = (BAND_B - BAND_T - SPLIT_GAP) / 2        # 115
UP_T, UP_B = BAND_T, BAND_T + ROW_H              # 70 .. 185
LO_T = BAND_B - ROW_H                            # 225 .. 340
Y_UP = UP_T + ROW_H / 2                          # 127.5  upper flow line
Y_LO = LO_T + ROW_H / 2                          # 282.5  lower flow line
Y_MID = (BAND_T + BAND_B) / 2                    # 205    main flow line

PAN_T, PAN_B = 400, 700                  # training panel
SUB_T, SUB_H = 444, 156                  # the four loss sub-blocks
SUB_N, SUB_GAP, SUB_PAD = 4, 24, 28
SUB_W = (RIGHT - M - 2 * SUB_PAD - (SUB_N - 1) * SUB_GAP) / SUB_N   # 329
SUB_X = [M + SUB_PAD + i * (SUB_W + SUB_GAP) for i in range(SUB_N)]

# ------------------------------------------------------------------ canvas
svg.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}" font-family="{FONT_STACK}" fill="{C_TXT}" '
    f'xml:space="preserve">')
svg.append('<defs>'
           '<marker id="ah" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{ARROW}"/></marker>'
           '<marker id="ahd" viewBox="0 0 10 10" refX="9.2" refY="5" markerWidth="7" '
           f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{NEW_S}"/></marker>'
           '</defs>')
svg.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')


def rule(x1, y, x2, colour="#DEE2E6", sw=1.2):
    svg.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
               f'stroke="{colour}" stroke-width="{sw}"/>')


# ------------------------------------------------------------ stage header
for cx, lbl in zip(CX, ("1 · INPUT", "2 · DUAL DETECTION", "3 · CONSOLIDATION",
                        "4 · TRACKING", "5 · SPATIO-TEMPORAL REASONING",
                        "6 · OUTPUT")):
    text(cx, 26, lbl, 12.5, "b", C_HEAD)
rule(M, 38, RIGHT)

# ---------------------------------------------------------- 1. video input
x, w, c = COL_X[0], COL_W[0], CX[0]
block(x, 130, w, 150, NEU_F, NEU_S)
text(c, 160, "Video input", 15, "b", box=w)
text(c, 184, "{*I*_{*t*}}, *t* = 1 … *T*", 12.5, box=w)
text(c, 212, "auto-orient", 11.5, fill=C_MUTE, box=w)
text(c, 232, "resize 640 × 640", 11.5, fill=C_MUTE, box=w)
text(c, 252, "adaptive contrast", 11.5, fill=C_MUTE, box=w)

# -------------------------------------------------- 2a. person detector
x, w, c = COL_X[1], COL_W[1], CX[1]
block(x, UP_T, w, ROW_H, STD_F, STD_S)
text(c, 99, "YOLOv12x — person detector", 16, "b", box=w)
text(c, 123, "off-the-shelf, COCO-pretrained, no fine-tuning", 12, box=w)
text(c, 145, "detected class: *person*", 12, box=w)
text(c, 167, "high recall and robustness in crowded scenes", 12, fill=C_MUTE, box=w)

# -------------------------------------------------- 2b. luggage detector
block(x, LO_T, w, ROW_H, NEW_F, NEW_S, sw=2.4)
text(c, 249, "YOLOv12m — luggage detector", 16, "b", box=w)
text(c, 271, "trained on the proposed public dataset", 12, box=w)
text(c, 289, "(29,053 images · 130,475 instances)", 12, box=w)
text(c, 307, "classes: backpack · bag · trolley", 12, box=w)
text(c, 328, "*network architecture unchanged*", 12, fill=NEW_D, box=w)

# ------------------------------------------------------ 3. consolidation
x, w, c = COL_X[2], COL_W[2], CX[2]
block(x, LO_T, w, ROW_H, STD_F, STD_S)
text(c, 254, "Class-agnostic NMS", 14, "b", box=w)
text(c, 280, "IoU > 0.5 → keep only", 11.5, box=w)
text(c, 298, "the highest-confidence", 11.5, box=w)
text(c, 316, "box, whatever its class", 11.5, box=w)
text(c, 335, "≤ 1 box per item / frame", 11, fill=C_MUTE, box=w)

# ----------------------------------------------------------- 4. tracking
x, w, c = COL_X[3], COL_W[3], CX[3]
block(x, BAND_T, w, BAND_B - BAND_T, STD_F, STD_S)
text(c, 100, "Tracking-by-detection", 15, "b", box=w)
rule(x + 22, 110, x + w - 22, "#B7C7DA")
text(c, 133, "constant-velocity motion", 11.5, box=w)
text(c, 151, "model → *b*_{τ}^{pred}", 11.5, box=w)
text(c, 180, "Hungarian global", 11.5, "b", box=w)
text(c, 197, "assignment", 11.5, "b", box=w)
text(c, 220, "*C*(*d*,τ) = −*w*_{IoU}·IoU(*b*_{*d*}, *b*_{τ}^{pred})", 10.5, box=w)
text(c, 238, "+ *w*_{dist}·||*c*_{*d*} − *b*_{τ}^{pred}||_{2} + λ_{cls}", 10.5, box=w)
text(c, 262, "λ_{cls} : soft class-mismatch penalty", 10, fill=C_MUTE, box=w)
rule(x + 22, 285, x + w - 22, "#B7C7DA")
text(c, 306, "each track carries a box,", 11, fill=C_MUTE, box=w)
text(c, 323, "a last-seen time and a velocity", 11, fill=C_MUTE, box=w)

# ---------------------------------------------------------- 5. reasoning
x, w, c = COL_X[4], COL_W[4], CX[4]
block(x, BAND_T, w, BAND_B - BAND_T, NEW_F, NEW_S, sw=2.4)
text(c, 93, "Abandonment reasoning", 15, "b", box=w)
ix, iw = x + 14, w - 28
for k, (top, title, l1, l2) in enumerate((
        (100, "(i) Proximity test",
         "a person within radius *R* of the luggage",
         "→ the nearest such person is the *owner*"),
        (180, "(ii) Owner hysteresis",
         "ownership switches only if a candidate",
         "stays closer for a confirmation time"),
        (260, "(iii) Unattended timer",
         "accrues only while the luggage is",
         "visible **and** unsupervised"))):
    block(ix, top, iw, 70, "#FFFFFF", NEW_S, sw=1.2, rx=6)
    text(c, top + 20, title, 12, "b", box=iw)
    text(c, top + 39, l1, 10.6, box=iw)
    text(c, top + 56, l2, 10.6, box=iw)

# ------------------------------------------------------------- 6. outputs
x, w, c = COL_X[5], COL_W[5], CX[5]
block(x, UP_T, w, ROW_H, ALM_F, ALM_S, sw=2.4)
text(c, 98, "UNATTENDED", 13, "b", ALM_T, box=w)
text(c, 116, "LUGGAGE", 13, "b", ALM_T, box=w)
text(c, 134, "ALARM", 13, "b", ALM_T, box=w)
rule(x + 20, 150, x + w - 20, "#E0B4AC")
text(c, 169, "*t*_{e} − *t*_{s} > *T*_{unattended}", 10, ALM_T, box=w)

block(x, LO_T, w, ROW_H, OK_F, OK_S)
text(c, 252, "Supervised", 14, "b", OK_T, box=w)
text(c, 274, "(no alarm)", 12, fill=OK_T, box=w)
rule(x + 20, 292, x + w - 20, "#B4D2BE")
text(c, 312, "a person stays within *R*", 10, OK_T, box=w)

# ----------------------------------------------------------------- arrows
e1 = COL_X[0] + COL_W[0]          # 178
j1 = e1 + GAP / 2                 # 196   split junction
arrow(f"M {e1} {Y_MID} L {j1} {Y_MID} L {j1} {Y_UP} L {COL_X[1] - 2} {Y_UP}")
arrow(f"M {e1} {Y_MID} L {j1} {Y_MID} L {j1} {Y_LO} L {COL_X[1] - 2} {Y_LO}")
arrow(f"M {COL_X[1] + COL_W[1]} {Y_UP} L {COL_X[3] - 2} {Y_UP}")
arrow(f"M {COL_X[1] + COL_W[1]} {Y_LO} L {COL_X[2] - 2} {Y_LO}")
arrow(f"M {COL_X[2] + COL_W[2]} {Y_LO} L {COL_X[3] - 2} {Y_LO}")
arrow(f"M {COL_X[3] + COL_W[3]} {Y_MID} L {COL_X[4] - 2} {Y_MID}")
e5 = COL_X[4] + COL_W[4]          # 1296
j5 = e5 + GAP / 2                 # 1314
arrow(f"M {e5} {Y_MID} L {j5} {Y_MID} L {j5} {Y_UP} L {COL_X[5] - 2} {Y_UP}")
arrow(f"M {e5} {Y_MID} L {j5} {Y_MID} L {j5} {Y_LO} L {COL_X[5] - 2} {Y_LO}")

text((CX[1] + CX[3]) / 2, Y_UP - 10, "person boxes", 10.5, fill=C_MUTE)
text((COL_X[1] + COL_W[1] + COL_X[2]) / 2, 210, "raw luggage boxes", 10.5, fill=C_MUTE)
text((COL_X[2] + COL_W[2] + COL_X[3]) / 2, 210, "1 box per item", 10.5, fill=C_MUTE)
text((COL_X[3] + COL_W[3] + COL_X[4]) / 2, Y_MID - 9, "tracks", 10.5, fill=C_MUTE)

# --------------------------------------------------------- training panel
arrow(f"M {CX[1]} {PAN_T} L {CX[1]} {BAND_B + 2}",
      stroke=NEW_S, sw=2.2, dash="7 5", marker="ahd")
text(CX[1] + 12, PAN_T - 27,
     "*supervises the training of the luggage detector only*", 11,
     fill=NEW_D, anchor="start")

block(M, PAN_T, RIGHT - M, PAN_B - PAN_T, "#FFFCF5", NEW_S, sw=2, rx=10, dash="8 5")
text((M + RIGHT) / 2, PAN_T + 29,
     "**Small-object–aware loss and assignment**   "
     "*(training time only — no inference cost, no architectural change)*", 14)

sx, sw_ = SUB_X, SUB_W
for i, (title, lines) in enumerate((
    ("(a) Dynamic curriculum weighting", [
        (500, "*w*_{*j*}(*t*) = α(*t*) *â*_{*j*} + (1 − α(*t*)) *s*_{*j*}", 12, C_TXT),
        (522, "*â*_{*j*} : normalised inverse area    *s*_{*j*} : target score", 11, C_TXT),
        (544, "α(*t*) = clip(0.9 − 0.4 *t*/*T*, 0.3, 0.9)", 11.5, C_TXT),
        (566, "reweights the IoU and DFL regression terms", 10.5, C_MUTE),
        (582, "early: area-dominant  →  late: balanced", 10.5, C_MUTE)]),
    ("(b) Auxiliary centre loss for small objects", [
        (500, "*L*_{center} = Σ_{*j*} ||*ĉ*_{*j*} − *c*_{*j*}|| · 1_{small}(*j*)", 12, C_TXT),
        (518, "(normalised by the number of small objects)", 10, C_MUTE),
        (542, "1_{small} : ground-truth area < 24 × 24 px", 11.5, C_TXT),
        (564, "λ_{center}(*t*) = max(0.01, 0.05 (1 − *t*/35))", 11.5, C_TXT),
        (582, "counters the IoU collapse on tiny boxes", 10.5, C_MUTE)]),
    ("(c) Adaptive per-batch loss clipping", [
        (500, "*L*_{IoU} ← min(*L*_{IoU}, *M*_{IoU}(*t*)),   *M*_{IoU}(*t*) = 10 + 10(1 − *t*/*T*)", 11, C_TXT),
        (522, "*L*_{DFL} ← min(*L*_{DFL}, *M*_{DFL}(*t*)),   *M*_{DFL}(*t*) = 5 + 5(1 − *t*/*T*)", 11, C_TXT),
        (546, "ceilings set by grid search, applied per batch", 11, C_TXT),
        (568, "suppresses the occasional loss spikes that", 10.5, C_MUTE),
        (582, "would otherwise destabilise optimisation", 10.5, C_MUTE)]),
    ("(d) Small-object-tuned label assignment", [
        (502, "task-aligned assigner with an enlarged", 11.5, C_TXT),
        (520, "candidate pool per ground-truth box", 11.5, C_TXT),
        (548, "**top-*k* :  10  →  25**", 16, C_TXT),
        (570, "more positive anchors per ground truth", 10.5, C_MUTE),
        (584, "→ fewer small-object false negatives", 10.5, C_MUTE)]),
)):
    block(sx[i], SUB_T, sw_, SUB_H, "#FFFFFF", NEW_S, sw=1.4, rx=7)
    cc = sx[i] + sw_ / 2
    text(cc, SUB_T + 24, title, 13, "b", box=sw_)
    rule(sx[i] + 18, SUB_T + 34, sx[i] + sw_ - 18, "#EBD3A8")
    for yy, mk, sz, col in lines:
        text(cc, yy, mk, sz, fill=col, box=sw_)

# combined objective
eq_t = 616
block(M + SUB_PAD, eq_t, RIGHT - M - 2 * SUB_PAD, 60, NEW_F, NEW_S, sw=1.6, rx=7)
text((M + RIGHT) / 2, eq_t + 28,
     "*L*(*t*) = λ_{box} *L*_{IoU} + λ_{DFL} *L*_{DFL} "
     "+ λ_{cls} *L*_{cls} + λ_{center}(*t*) *L*_{center}", 17)
text((M + RIGHT) / 2, eq_t + 50,
     "λ_{box}, λ_{DFL} and λ_{cls} keep their baseline YOLOv12 values; only "
     "λ_{center}(*t*) and the weighting / clipping schedules are introduced here",
     11, fill=C_MUTE)

# ----------------------------------------------------------------- legend
LEG_Y = PAN_B + 32
slot = (RIGHT - M) / 4
for i, (f, s, sw_l, dash, lbl) in enumerate((
        (NEW_F, NEW_S, 2, None, "proposed contribution (this work)"),
        (STD_F, STD_S, 1.6, None, "standard / off-the-shelf component"),
        ("#FFFCF5", NEW_S, 1.8, "5 3", "active at training time only"),
        (ALM_F, ALM_S, 2, None, "alarm event"))):
    lx = M + i * slot
    block(lx, LEG_Y - 12, 24, 15, f, s, sw=sw_l, rx=3, dash=dash)
    text(lx + 32, LEG_Y, lbl, 11.5, fill=C_MUTE, anchor="start")

svg.append("</svg>")

with open(OUT_SVG, "w", encoding="utf-8") as fh:
    fh.write("\n".join(svg) + "\n")

try:
    import cairosvg
    cairosvg.svg2png(url=OUT_SVG, write_to=OUT_PNG,
                     output_width=int(W * PNG_SCALE), output_height=int(H * PNG_SCALE),
                     background_color="white")
except ImportError:
    print("cairosvg not installed — SVG written, PNG skipped")

print(f"wrote {OUT_SVG}")
print(f"wrote {OUT_PNG}")
print(f"columns x = {[round(v) for v in COL_X]}  right edge = {RIGHT}")
if warn:
    print("\ntext that does not fit its box:")
    print("\n".join(warn))
else:
    print("all text fits inside its box")
