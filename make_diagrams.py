"""
Generate the architecture diagrams embedded in the model READMEs:

    lenet5/lenet5.png
    alexnet/alexnet.png

    python make_diagrams.py

Nothing is hand-entered: parameter counts and output shapes are read off the
models themselves (forward hooks on a dummy input), and LeNet-5's C3 grid is
drawn from C3_CONNECTIONS. If the code changes, re-run this and the diagrams
follow.

Each layer is one row: a box with its name, operation and output shape, and a
bar whose length is LINEAR in its parameter count. On a linear scale AlexNet's
fc1 runs nearly the full width and the conv layers are slivers -- which is the
point.

Output is 3840 x 2160 (16 x 9 in at 240 dpi).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "lenet5"), str(ROOT / "alexnet")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import torch

from lenet5 import LeNet5, C3_CONNECTIONS
from alexnet import AlexNet


# ============================================================
# STYLE
#   Three layer-type hues, validated as a set (all pairs, incl. CVD).
#   Text always uses the ink tokens, never a series colour.
# ============================================================

SURFACE = "#fcfcfb"
PANEL = "#f0efec"
RULE = "#dcdbd5"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"

CONV = "#2a78d6"
FC = "#eb6834"
SUBSAMPLE = "#1baf7a"
KIND_COLOUR = {"conv": CONV, "fc": FC, "sub": SUBSAMPLE, None: None}
KIND_LABEL = {"conv": "convolution", "fc": "fully connected", "sub": "trainable subsampling"}

FIGSIZE = (16, 9)
DPI = 240

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# ============================================================
# MEASURING THE MODELS
# ============================================================

def output_shapes(model, x, names):
    """Output shape (without batch dim) of each named submodule."""
    shapes, hooks = {}, []
    for name in names:
        def hook(_module, _inputs, output, name=name):
            shapes[name] = tuple(output.shape[1:])
        hooks.append(model.get_submodule(name).register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return shapes


def n_params(model, name):
    return sum(p.numel() for p in model.get_submodule(name).parameters())


def fmt_shape(shape):
    return " × ".join(str(d) for d in shape)


# ============================================================
# DRAWING
# ============================================================

BOX_X0, BOX_X1 = 0.0, 40.0      # layer box
BAR_X0, BAR_X1 = 42.0, 88.0     # parameter bar region (linear scale)


def draw_layers(ax, rows, total, box_x1=BOX_X1, bar_x0=BAR_X0, bar_x1=BAR_X1):
    """
    rows: dicts with name, op, shape, params, kind, and optional
          tag (pill text inside the box) and emphasis (for param-free rows
          that matter, e.g. LRN).
    """
    BOX_X1, BAR_X0, BAR_X1 = box_x1, bar_x0, bar_x1
    n = len(rows)
    ax.set_xlim(0, 100)
    ax.set_ylim(n - 0.4, -1.3)
    ax.axis("off")

    biggest = max(r["params"] for r in rows)

    ax.text(BOX_X0 + 1.0, -0.95, "Layer", color=INK_2, fontsize=11, va="center")
    ax.text(BOX_X1 - 1.0, -0.95, "Output shape", color=INK_2, fontsize=11,
            va="center", ha="right")
    ax.text(BAR_X0, -0.95, "Parameters  (bar length to scale)", color=INK_2,
            fontsize=11, va="center")
    ax.plot([BOX_X0, 100], [-0.6, -0.6], color=RULE, lw=1)

    for y, r in enumerate(rows):
        colour = KIND_COLOUR[r["kind"]]
        trainable = r["kind"] is not None

        # --- layer box ------------------------------------------------
        h = 0.78 if trainable else 0.52
        ax.add_patch(FancyBboxPatch(
            (BOX_X0 + 0.2, y - h / 2), BOX_X1 - BOX_X0 - 0.4, h,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=PANEL if trainable else SURFACE,
            edgecolor=RULE, lw=1, mutation_aspect=0.06,
        ))
        if trainable:      # type stripe -- identity is also in the op text
            ax.add_patch(Rectangle((BOX_X0 + 0.2, y - h / 2), 0.55, h,
                                   facecolor=colour, edgecolor="none"))

        name_ink = INK if (trainable or r.get("emphasis")) else MUTED
        ax.text(BOX_X0 + 1.4, y - (0.12 if trainable else 0), r["name"],
                color=name_ink, fontsize=13 if trainable else 11,
                fontweight="bold", va="center")
        op_x = BOX_X0 + 1.4 if trainable else BOX_X0 + 7.0
        op_y = y + 0.2 if trainable else y
        ax.text(op_x, op_y, r["op"], color=INK_2 if (trainable or r.get("emphasis")) else MUTED,
                fontsize=10, va="center")
        ax.text(BOX_X1 - 1.0, y - (0.12 if trainable else 0), r["shape"],
                color=INK if trainable else MUTED, fontsize=11.5 if trainable else 10,
                va="center", ha="right", family="DejaVu Sans Mono")

        if r.get("tag"):
            ax.text(BOX_X1 - 1.0, y + 0.2, r["tag"], color=INK_2, fontsize=9,
                    va="center", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE,
                              edgecolor=INK_2, lw=0.8))

        # --- parameter bar -------------------------------------------
        if not trainable:
            continue
        w = (BAR_X1 - BAR_X0) * r["params"] / biggest
        ax.add_patch(Rectangle((BAR_X0, y - 0.22), w, 0.44,
                               facecolor=colour, edgecolor="none"))
        share = 100 * r["params"] / total
        share_txt = f"{share:.1f}%" if share >= 0.1 else f"{share:.2f}%"
        ax.text(BAR_X0 + w + 0.6, y, f"{r['params']:,}   ({share_txt})", color=INK,
                fontsize=12, va="center")


def draw_header(fig, title, subtitle, kinds):
    fig.text(0.02, 0.955, title, color=INK, fontsize=20, fontweight="bold", va="center")
    fig.text(0.02, 0.915, subtitle, color=INK_2, fontsize=12.5, va="center")

    # legend: swatch + label, right-aligned row
    x = 0.98
    for kind in reversed(kinds):
        label = KIND_LABEL[kind]
        t = fig.text(x, 0.955, label, color=INK_2, fontsize=11.5, va="center", ha="right")
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(fig.transFigure.inverted())
        fig.patches.append(Rectangle((bb.x0 - 0.016, 0.947), 0.011, 0.016,
                                     transform=fig.transFigure, figure=fig,
                                     facecolor=KIND_COLOUR[kind], edgecolor="none"))
        x = bb.x0 - 0.03


def footer(fig, text):
    fig.text(0.02, 0.025, text, color=MUTED, fontsize=10, va="center")


# ============================================================
# LENET-5
# ============================================================

def lenet5_diagram(path):
    model = LeNet5()
    names = ["c1", "s2", "c3", "s4", "c5", "f6", "rbf"]
    shapes = output_shapes(model, torch.zeros(1, 1, 32, 32), names)
    p = {n: n_params(model, n) for n in names}
    total = sum(p.values())

    rows = [
        dict(name="Input", op="MNIST, padded to 32×32, in [−0.1, 1.175]",
             shape="1 × 32 × 32", params=0, kind=None, emphasis=True),
        dict(name="C1", op="5×5 conv, 6 maps, scaled tanh",
             shape=fmt_shape(shapes["c1"]), params=p["c1"], kind="conv"),
        dict(name="S2", op="2×2 sum × trainable coeff + bias, tanh",
             shape=fmt_shape(shapes["s2"]), params=p["s2"], kind="sub"),
        dict(name="C3", op="5×5 conv, 16 maps, partial (Table I, right)",
             shape=fmt_shape(shapes["c3"]), params=p["c3"], kind="conv"),
        dict(name="S4", op="2×2 sum × trainable coeff + bias, tanh",
             shape=fmt_shape(shapes["s4"]), params=p["s4"], kind="sub"),
        dict(name="C5", op="5×5 conv, 120 maps (1×1 output)",
             shape=fmt_shape(shapes["c5"]), params=p["c5"], kind="conv"),
        dict(name="F6", op="fully connected, 84 units (= 7×12 bitmap)",
             shape=fmt_shape(shapes["f6"]), params=p["f6"], kind="fc"),
        dict(name="RBF", op="distance to 10 FROZEN prototypes, argmin",
             shape=fmt_shape(shapes["rbf"]), params=0, kind=None, emphasis=True),
    ]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], left=0.02, right=0.98,
                          top=0.86, bottom=0.07, wspace=0.06)
    draw_layers(fig.add_subplot(gs[0]), rows, total, box_x1=54.0, bar_x0=56.0, bar_x1=86.0)
    c5_share = 100 * p["c5"] / total
    draw_header(fig, f"LeNet-5  ·  LeCun et al., 1998  ·  {total:,} parameters",
                f"C5 alone holds {c5_share:.1f}% of the parameters. The RBF prototypes "
                f"are fixed, not trained, so the output layer adds none.",
                ["conv", "sub", "fc"])

    # --- C3 connection grid (Table I): rows = S2 maps, cols = C3 maps ---
    ax = fig.add_subplot(gs[1])
    ax.set_aspect("equal")
    ax.axis("off")
    cell = 0.84
    for j, conn in enumerate(C3_CONNECTIONS):
        for i in range(6):
            on = i in conn
            ax.add_patch(Rectangle((j + (1 - cell) / 2, i + (1 - cell) / 2), cell, cell,
                                   facecolor=CONV if on else PANEL, edgecolor="none"))
    for j in range(16):
        ax.text(j + 0.5, -0.35, str(j), ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.text(j + 0.5, 6.35, str(len(C3_CONNECTIONS[j])), ha="center", va="top",
                fontsize=8.5, color=INK_2)
    for i in range(6):
        ax.text(-0.3, i + 0.5, str(i), ha="right", va="center", fontsize=8.5, color=INK_2)
    ax.text(8, -1.45, "C3 feature map", ha="center", fontsize=10, color=INK_2)
    ax.text(-1.35, 3, "S2 feature map", ha="center", va="center", rotation=90,
            fontsize=10, color=INK_2)
    ax.text(-0.3, 6.35, "inputs", ha="right", va="top", fontsize=8.5, color=MUTED)

    # brackets for the four groups
    for a, b, label in [(0, 6, "3 contiguous"), (6, 12, "4 contiguous"),
                        (12, 15, "4 non-\ncontiguous"), (15, 16, "all")]:
        ax.plot([a + 0.1, a + 0.1, b - 0.1, b - 0.1], [7.05, 7.25, 7.25, 7.05],
                color=INK_2, lw=0.9)
        ax.text((a + b) / 2, 7.45, label, ha="center", va="top", fontsize=8.5, color=INK_2)

    links = sum(len(c) for c in C3_CONNECTIONS)
    ax.text(0, -2.6, "C3 connection table (paper, Table I)", fontsize=13,
            fontweight="bold", color=INK)
    ax.text(0, 9.0, f"{links} of 96 possible S2→C3 links.\n"
                    f"{p['c3']:,} parameters instead of {16 * (6 * 25 + 1):,} for full "
                    f"connection.\nEach S2 map feeds exactly "
                    f"{sum(0 in c for c in C3_CONNECTIONS)} C3 maps.",
            fontsize=10, color=INK_2, va="top", linespacing=1.5)
    ax.set_xlim(-2, 16.2)
    ax.set_ylim(11.2, -3.2)

    footer(fig, "Generated by make_diagrams.py from lenet5/lenet5.py. "
                "Counts and shapes are measured from the model, not typed in.")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return total


# ============================================================
# ALEXNET
# ============================================================

def alexnet_diagram(path):
    model = AlexNet()
    f, c = "features.", "classifier."
    names = [f + n for n in ("conv1", "lrn1", "pool1", "conv2", "lrn2", "pool2",
                             "conv3", "conv4", "conv5", "pool5")] + \
            [c + n for n in ("fc1", "fc2", "fc3")]
    shapes = output_shapes(model, torch.zeros(1, 3, 224, 224), names)
    total = sum(p.numel() for p in model.parameters())

    def conv_row(n, op, tag=None):
        mod = model.get_submodule(f + n)
        if mod.groups > 1:
            tag = f"groups={mod.groups} · GPU split"
        return dict(name=n, op=op, shape=fmt_shape(shapes[f + n]),
                    params=n_params(model, f + n), kind="conv", tag=tag)

    def lrn_row(n):
        return dict(name="LRN", op="response normalisation, BEFORE max-pool (§3.5)",
                    shape=fmt_shape(shapes[f + n]), params=0, kind=None, emphasis=True)

    def pool_row(n):
        return dict(name="pool", op="max-pool 3×3, stride 2 (overlapping)",
                    shape=fmt_shape(shapes[f + n]), params=0, kind=None)

    def fc_row(n, op):
        return dict(name=n, op=op, shape=fmt_shape(shapes[c + n]),
                    params=n_params(model, c + n), kind="fc")

    rows = [
        dict(name="Input", op="RGB crop", shape="3 × 224 × 224", params=0, kind=None,
             emphasis=True),
        conv_row("conv1", "11×11 stride 4, 96 maps, ReLU"), lrn_row("lrn1"), pool_row("pool1"),
        conv_row("conv2", "5×5, 256 maps, ReLU"), lrn_row("lrn2"), pool_row("pool2"),
        conv_row("conv3", "3×3, 384 maps, ReLU", tag="sees both GPUs"),
        conv_row("conv4", "3×3, 384 maps, ReLU"),
        conv_row("conv5", "3×3, 256 maps, ReLU"), pool_row("pool5"),
        fc_row("fc1", "9216 → 4096, ReLU, dropout on output"),
        fc_row("fc2", "4096 → 4096, ReLU, dropout on output"),
        fc_row("fc3", "4096 → 1000 logits"),
    ]

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.79])
    draw_layers(ax, rows, total)

    fc1 = n_params(model, c + "fc1")
    conv_total = sum(r["params"] for r in rows if r["kind"] == "conv")
    draw_header(fig, f"AlexNet  ·  Krizhevsky et al., 2012  ·  {total:,} parameters",
                f"fc1 alone holds {100 * fc1 / total:.1f}% of the parameters; all five conv "
                f"layers together hold {100 * conv_total / total:.1f}%.",
                ["conv", "fc"])
    footer(fig, "Generated by make_diagrams.py from alexnet/alexnet.py. "
                "Counts and shapes are measured from the model, not typed in.")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return total


if __name__ == "__main__":
    for name, draw in [("lenet5", lenet5_diagram), ("alexnet", alexnet_diagram)]:
        out = ROOT / name / f"{name}.png"
        total = draw(out)
        print(f"wrote {out.relative_to(ROOT)}  ({total:,} parameters)")
