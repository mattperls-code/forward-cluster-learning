"""
arch_diagram.py — Neural architecture diagram generator
"""

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# ── colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "conv":   {"fill": "#EEEDFE", "stroke": "#534AB7", "text": "#3C3489"},
    "linear": {"fill": "#E1F5EE", "stroke": "#0F6E56", "text": "#085041"},
    "pred":   {"fill": "#FAEEDA", "stroke": "#BA7517", "text": "#854F0B"},
    "pool":   {"fill": "#F1EFE8", "stroke": "#5F5E5A", "text": "#444441"},
    "output": {"fill": "#FAECE7", "stroke": "#993C1D", "text": "#712B13"},
    "misc":   {"fill": "#F1EFE8", "stroke": "#5F5E5A", "text": "#444441"},
}

DARK = {
    "conv":   {"fill": "#3C3489", "stroke": "#AFA9EC", "text": "#CECBF6"},
    "linear": {"fill": "#085041", "stroke": "#5DCAA5", "text": "#9FE1CB"},
    "pred":   {"fill": "#633806", "stroke": "#EF9F27", "text": "#FAC775"},
    "pool":   {"fill": "#444441", "stroke": "#B4B2A9", "text": "#D3D1C7"},
    "output": {"fill": "#4A1B0C", "stroke": "#F0997B", "text": "#F5C4B3"},
    "misc":   {"fill": "#444441", "stroke": "#B4B2A9", "text": "#D3D1C7"},
}

# ── layer descriptor ─────────────────────────────────────────────────────────
@dataclass
class Layer:
    kind: str
    label: str
    sublabel: str = ""
    is_pred: bool = False

# ── SVG helpers ──────────────────────────────────────────────────────────────
NS = "http://www.w3.org/2000/svg"

def el(tag, **attrs):
    e = ET.Element(tag)
    for k, v in attrs.items():
        e.set(k.replace("_", "-"), str(v))
    return e

def text_el(content, x, y, cls):
    t = ET.Element("text")
    t.set("x", str(x))
    t.set("y", str(y))
    t.set("text-anchor", "middle")
    t.set("dominant-baseline", "central")
    t.set("class", cls)
    t.text = content
    return t

# ── layout constants ─────────────────────────────────────────────────────────
BOX_W = 200
BOX_H = 54
BOX_H_SM = 36
PRED_W = 68
GAP = 14
PRED_GAP = 8

# ── draw column ──────────────────────────────────────────────────────────────
def draw_column(layers, cx, start_y, dark=False):
    pal = DARK if dark else PALETTE
    g = ET.Element("g")
    y = start_y

    last_main_y = None
    last_main_h = None

    for layer in layers:
        c = pal[layer.kind]
        h = BOX_H_SM if (not layer.sublabel and layer.kind in ("misc", "pool")) else BOX_H

        if layer.is_pred:
            if last_main_y is None:
                continue  # safety

            # attach to previous main layer center
            py = last_main_y + last_main_h // 2 - h // 2
            px = cx + BOX_W // 2 + PRED_GAP

            # arrow (black)
            g.append(el("line",
                x1=cx + BOX_W // 2 - 2, y1=last_main_y + last_main_h // 2,
                x2=px - 2,             y2=last_main_y + last_main_h // 2,
                stroke="#000", stroke_width="1",
                marker_end="url(#arr)"
            ))

            # box
            g.append(el("rect",
                x=px, y=py, width=PRED_W, height=h, rx="8",
                fill=c["fill"], stroke=c["stroke"], stroke_width="0.8"
            ))

            # text
            g.append(text_el(layer.label, px + PRED_W // 2, py + h // 2 - 6, "lbl"))
            if layer.sublabel:
                g.append(text_el(layer.sublabel, px + PRED_W // 2, py + h // 2 + 9, "sub"))

            continue

        # main box
        g.append(el("rect",
            x=cx - BOX_W // 2, y=y, width=BOX_W, height=h, rx="8",
            fill=c["fill"], stroke=c["stroke"], stroke_width="0.8"
        ))

        g.append(text_el(layer.label, cx, y + h // 2 - 7, "lbl"))
        if layer.sublabel:
            g.append(text_el(layer.sublabel, cx, y + h // 2 + 8, "sub"))

        # store anchor point
        last_main_y = y
        last_main_h = h

        y += h + GAP

    return g

# ── arrows between main boxes ────────────────────────────────────────────────
def add_arrows(layers, cx, start_y):
    g = ET.Element("g")
    y = start_y
    prev_bottom = None

    for layer in layers:
        h = BOX_H_SM if (not layer.sublabel and layer.kind in ("misc", "pool")) else BOX_H

        if not layer.is_pred:
            if prev_bottom is not None:
                g.append(el("line",
                    x1=cx, y1=prev_bottom,
                    x2=cx, y2=y,
                    stroke="#000", stroke_width="1.2",
                    marker_end="url(#arr)"
                ))
            prev_bottom = y + h
            y += h + GAP

    return g

# ── build svg ────────────────────────────────────────────────────────────────
def build_svg(bp_layers, lpsl_layers, bp_title, lpsl_title, dark=False):
    W = 600 + PRED_W
    MARGIN_TOP = 52
    COL1 = 170
    COL2 = 420

    def height(layers):
        y = 0
        for l in layers:
            if not l.is_pred:
                h = BOX_H_SM if (not l.sublabel and l.kind in ("misc", "pool")) else BOX_H
                y += h + GAP
        return y - GAP

    content_h = max(height(bp_layers), height(lpsl_layers))
    H = MARGIN_TOP + content_h + 40

    svg = el("svg", width="100%", viewBox=f"0 0 {W} {H}", xmlns=NS)

    # defs
    defs = ET.SubElement(svg, "defs")
    marker = ET.SubElement(defs, "marker",
        id="arr", viewBox="0 0 10 10", refX="8", refY="5",
        markerWidth="6", markerHeight="6", orient="auto"
    )
    ET.SubElement(marker, "path",
        d="M2 1L8 5L2 9",
        fill="none", stroke="#000",
        **{"stroke-width": "1.5"}
    )

    # styles
    style = ET.SubElement(svg, "style")
    style.text = """
.title { font-size: 18px; font-weight: 700; fill: #222; }
.lbl   { font-size: 12px; font-weight: 600; }
.sub   { font-size: 10px; }
.border { fill: none; stroke: #000; stroke-width: 1; stroke-dasharray: 4 3; }
"""

    
    pad = 12

    # left column bounds
    left_x = COL1 - BOX_W // 2 - pad
    left_w = BOX_W + 2 * pad

    # right column bounds (include prediction heads fully)
    right_x = COL2 - BOX_W // 2 - pad
    right_w = BOX_W + 2 * pad + PRED_GAP + PRED_W  # +6 = safety for stroke/arrow tip

    svg.append(el("rect",
        x=left_x, y=MARGIN_TOP - 10,
        width=left_w,
        height=height(bp_layers) + 20,
        rx="12", **{"class": "border"}
    ))

    svg.append(el("rect",
        x=right_x, y=MARGIN_TOP - 10,
        width=right_w,
        height=height(lpsl_layers) + 20,
        rx="12", **{"class": "border"}
    ))

    # titles
    svg.append(text_el(bp_title, COL1, 24, "title"))
    svg.append(text_el(lpsl_title, COL2 + 30, 24, "title"))

    # columns
    svg.append(add_arrows(bp_layers, COL1, MARGIN_TOP))
    svg.append(draw_column(bp_layers, COL1, MARGIN_TOP, dark))

    svg.append(add_arrows(lpsl_layers, COL2, MARGIN_TOP))
    svg.append(draw_column(lpsl_layers, COL2, MARGIN_TOP, dark))

    return ET.tostring(svg, encoding="unicode")

# ── examples ─────────────────────────────────────────────────────────────────

# def example_bp():
#     return [
#         Layer("conv",   "Conv + ReLU ×2 + MaxPool", "1×28×28 → 32×14×14"),
#         Layer("conv",   "Conv + ReLU ×2 + MaxPool", "32×14×14 → 64×7×7"),
#         Layer("misc",   "Flatten",                  "64×7×7 → 3136"),
#         Layer("linear", "Linear + ReLU",            "3136 → 1024"),
#         Layer("linear", "Linear + ReLU",            "1024 → 128"),
#         Layer("output", "Linear + Softmax",                   "128 → 10"),
#     ]


# def example_lpsl():
#     return [
#         # Block 1
#         Layer("conv", "Conv + ReLU ×2 + MaxPool", "1×28×28 → 32×14×14"),
#         Layer("pred", "Prediction", "3136 → 10", True),

#         # Block 2
#         Layer("conv", "Conv + ReLU ×2 + MaxPool", "32×14×14 → 64×7×7"),
#         Layer("pred", "Prediction", "3136 → 10", True),

#         # Dense segment 1
#         Layer("misc",   "Flatten",               "64×7×7 → 3136"),
#         Layer("linear", "Linear + ReLU",         "3136 → 1024"),
#         Layer("pred",   "Prediction",            "1024 → 10", True),

#         # Dense segment 2
#         Layer("linear", "Linear + ReLU",         "1024 → 128"),
#         Layer("pred",   "Prediction",            "128 → 10", True)
#     ]


# def example_bp():
#     return [
#         Layer("linear", "Linear + ReLU", "64 → 32"),
#         Layer("linear", "Linear + ReLU", "32 → 32"),
#         Layer("linear", "Linear + ReLU", "32 → 32"),
#         Layer("output", "Linear + Softmax",        "32 → 5"),
#     ]


# def example_lpsl():
#     return [
#         # Segment 1
#         Layer("linear", "Linear + ReLU", "64 → 32"),
#         Layer("pred",   "Prediction",    "32 → 5", True),

#         # Segment 2
#         Layer("linear", "Linear + ReLU", "32 → 32"),
#         Layer("pred",   "Prediction",    "32 → 5", True),

#         # Segment 3
#         Layer("linear", "Linear + ReLU", "32 → 32"),
#         Layer("pred",   "Prediction",    "32 → 5", True)
#     ]

def example_bp():
    return [
        Layer("linear", "Embedding + Positional Encoding", "(31 × SequenceLen) → (128 × SequenceLen)"),

        Layer("conv", "Transformer Encoder ×3", "(128 × SequenceLen) → (128 × SequenceLen)"),

        Layer("pool", "MeanToken", "(128 × SequenceLen) → 128"),

        Layer("linear", "Linear + ReLU", "128 → 64"),

        Layer("output", "Linear + Softmax", "64 → 31"),
    ]


def example_lpsl():
    return [
        Layer("linear", "Embedding + Positional Encoding", "(31 × SequenceLen) → (128 × SequenceLen)"),

        Layer("conv", "Transformer Encoder ×2", "(128 × SequenceLen) → (128 × SequenceLen)"),
        Layer("pred", "Prediction", "128 → 31", True),

        Layer("conv", "Transformer Encoder ×1", "(128 × SequenceLen) → (128 × SequenceLen)"),
        Layer("pred", "Prediction", "128 → 31", True),

        Layer("pool", "MeanToken", "(128 × SequenceLen) → 128"),
        Layer("linear", "Linear + ReLU", "128 → 64"),
        Layer("pred", "Prediction", "64 → 31", True)
    ]

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="diagrams/transformer.svg")
    args = parser.parse_args()

    svg = build_svg(
        example_bp(),
        example_lpsl(),
        "Backpropagation Model",
        "LPSL Model"
    )

    with open(args.out, "w") as f:
        f.write(svg)

    print("Saved →", args.out)

if __name__ == "__main__":
    main()