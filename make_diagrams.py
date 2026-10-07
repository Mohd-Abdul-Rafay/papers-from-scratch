"""
Generate the architecture diagrams embedded in the model READMEs and on the
portfolio's paper pages:

    lenet5/lenet5.png    lenet5/lenet5.svg
    alexnet/alexnet.png  alexnet/alexnet.svg

    python make_diagrams.py

Nothing is hand-entered: parameter counts and output shapes are read off the
models themselves (forward hooks on a dummy input), and LeNet-5's C3 grid is
drawn from C3_CONNECTIONS. If the code changes, re-run this and the diagrams
follow.

Each layer is one card: name, operation and output shape on the first line,
and -- for layers with parameters -- a bar on the second line whose length is
LINEAR in its parameter count. On a linear scale AlexNet's fc1 runs nearly the
full width and the conv layers are slivers -- which is the point.

The figure is 8 in wide and as tall as its content: a portrait shape meant to
sit in an article column at web width without the rows shrinking to nothing.
PNG is 240 dpi (1920 px wide), for the READMEs and slides; SVG is for the
site. Colours are the portfolio's CSS tokens. Labels use Inter Tight and
shapes/counts JetBrains Mono; if either is missing the run still completes
on matplotlib's defaults, but prints a loud warning (see pick_font).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "lenet5"), str(ROOT / "alexnet")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import torch

from lenet5 import LeNet5, C3_CONNECTIONS
from alexnet import AlexNet


# ============================================================
# STYLE -- portfolio tokens (rafay-portfolio/src/app/globals.css)
#   Layer-type hue is carried by fills only; text always uses the ink
#   tokens, never a series colour.
# ============================================================

PAPER = "#F9F6EE"     # --paper   figure background
CARD = "#FFFDF8"      # --card    layer cards
RULE = "#D6D5D1"      # --hair    (ink at 15% over paper, flattened)
INK = "#0F172A"       # --ink
INK_2 = "#64706B"     # --muted   secondary text
FAINT = "#9AA29E"     # muted, lightened -- parameter-free layers
EMPTY = "#EBE9E2"     # unconnected C3 cell (ink at 6% over paper)

CONV = "#2563EB"      # --blue
FC = "#D3542A"        # --rust
SUBSAMPLE = "#6B8F73" # --sage
KIND_COLOUR = {"conv": CONV, "fc": FC, "sub": SUBSAMPLE, None: None}
KIND_LABEL = {"conv": "convolution", "fc": "fully connected", "sub": "trainable subsampling"}


# A missing font doesn't fail the run -- matplotlib's default is used
# instead -- but it changes every glyph path in the SVGs, so a
# regeneration on a machine without these fonts shows up as a large
# "mystery" diff. pick_font() makes that impossible to miss.
# Install (both SIL OFL): Inter Tight from fonts.google.com, JetBrains
# Mono from github.com/JetBrains/JetBrainsMono, then delete matplotlib's
# fontlist-*.json cache (python -c "import matplotlib; print(matplotlib.get_cachedir())").

MISSING_FONTS = []


def pick_font(preferred, fallback):
    installed = {f.name for f in font_manager.fontManager.ttflist}
    if preferred in installed:
        return preferred
    MISSING_FONTS.append((preferred, fallback))
    return fallback


def warn_missing_fonts():
    if not MISSING_FONTS:
        return
    rule = "!" * 72
    lines = [f"  {want!r} is not installed -- falling back to {got!r}."
             for want, got in MISSING_FONTS]
    print("\n".join([
        rule,
        "  WARNING: DIAGRAM FONTS MISSING",
        *lines,
        "  The PNG/SVG output WILL differ from the committed files. Do not",
        "  commit these diagrams; install the fonts (see top of file) first.",
        rule,
    ]), file=sys.stderr)


SANS = pick_font("Inter Tight", "DejaVu Sans")
MONO = pick_font("JetBrains Mono", "DejaVu Sans Mono")

DPI = 240
W = 8.0               # figure width, inches; height follows the content
M = 0.35              # side margin
NAME_X = M + 0.22     # layer name
OP_X = M + 1.05       # operation text, and the start of the parameter bar
SHAPE_X = W - M - 0.18

plt.rcParams.update({
    "font.family": SANS,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "svg.hashsalt": "papers-from-scratch",   # stable ids -> stable diffs
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
#   One axes covering the whole figure, in inches, y pointing down.
#   Everything is placed with a y cursor; the figure height is set from
#   the cursor at the end (see finish()).
# ============================================================

def new_canvas():
    fig = plt.figure(figsize=(W, 1), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax


def text_width(fig, s, size, family=SANS, weight="normal"):
    t = fig.text(0, 0, s, fontsize=size, family=family, fontweight=weight)
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    t.remove()
    return w


def wrap(fig, s, size, max_w):
    """Greedy word wrap to max_w inches, measured with the real font."""
    lines = []
    for paragraph in s.split("\n"):
        words, line = paragraph.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and text_width(fig, trial, size) > max_w:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def draw_header(fig, ax, y, title, subtitle, kinds):
    ax.text(M, y, title, color=INK, fontsize=15, fontweight="bold", va="top")
    y += 0.36
    for line in wrap(fig, subtitle, 10, W - 2 * M):
        ax.text(M, y, line, color=INK_2, fontsize=10, va="top")
        y += 0.2
    y += 0.12

    # legend: swatch + label, left-aligned row
    x = M
    for kind in kinds:
        ax.add_patch(Rectangle((x, y + 0.02), 0.13, 0.13,
                               facecolor=KIND_COLOUR[kind], edgecolor="none"))
        label = KIND_LABEL[kind]
        ax.text(x + 0.2, y + 0.085, label, color=INK_2, fontsize=9.5, va="center")
        x += 0.2 + text_width(fig, label, 9.5) + 0.35
    return y + 0.45


def draw_layers(fig, ax, y, rows, total):
    """
    rows: dicts with name, op, shape, params, kind, and optional
          tag (pill text on the bar line) and emphasis (for param-free rows
          that matter, e.g. LRN).
    Returns the y below the last row.
    """
    biggest = max(r["params"] for r in rows)
    longest_label = max(text_width(fig, bar_label(r, total), 9.5, MONO)
                        for r in rows if r["kind"])
    bar_max = SHAPE_X - OP_X - longest_label - 0.15

    ax.text(NAME_X, y, "Layer", color=INK_2, fontsize=9, va="bottom")
    ax.text(OP_X, y, "Operation  ·  parameters (bar length to scale)", color=INK_2,
            fontsize=9, va="bottom")
    ax.text(SHAPE_X, y, "Output shape", color=INK_2, fontsize=9, va="bottom", ha="right")
    y += 0.08
    ax.plot([M, W - M], [y, y], color=RULE, lw=0.8)
    y += 0.1

    for r in rows:
        colour = KIND_COLOUR[r["kind"]]
        trainable = r["kind"] is not None
        loud = trainable or r.get("emphasis")

        if trainable:
            h = 0.62
            ax.add_patch(FancyBboxPatch(
                (M, y), W - 2 * M, h,
                boxstyle="round,pad=0,rounding_size=0.05",
                facecolor=CARD, edgecolor=RULE, lw=0.8,
            ))
            ax.add_patch(Rectangle((M, y), 0.06, h, facecolor=colour, edgecolor="none"))
            line_a = y + 0.2
        else:
            h = 0.36
            line_a = y + h / 2

        ax.text(NAME_X, line_a, r["name"], color=INK if loud else FAINT,
                fontsize=12 if trainable else 10.5, fontweight="bold", va="center")
        ax.text(OP_X, line_a, r["op"], color=INK_2 if loud else FAINT,
                fontsize=9.5, va="center")
        ax.text(SHAPE_X, line_a, r["shape"], color=INK if trainable else FAINT,
                fontsize=10 if trainable else 9, va="center", ha="right", family=MONO)

        if trainable:
            line_b = y + 0.44
            # To scale, but never thinner than a hairline: S2's 12 parameters
            # would otherwise round to nothing and read as "no bar".
            w = max(bar_max * r["params"] / biggest, 0.015)
            ax.add_patch(Rectangle((OP_X, line_b - 0.065), w, 0.13,
                                   facecolor=colour, edgecolor="none"))
            ax.text(OP_X + w + 0.08, line_b, bar_label(r, total), color=INK,
                    fontsize=9.5, va="center", family=MONO)
            if r.get("tag"):
                ax.text(SHAPE_X, line_b, r["tag"], color=INK_2, fontsize=8.5,
                        va="center", ha="right",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=CARD,
                                  edgecolor=INK_2, lw=0.6))

        y += h + 0.08
    return y


def bar_label(r, total):
    share = 100 * r["params"] / total
    share_txt = f"{share:.1f}%" if share >= 0.1 else f"{share:.2f}%"
    return f"{r['params']:,}  ({share_txt})"


def finish(fig, ax, y, footer_text, path):
    y += 0.1
    for line in wrap(fig, footer_text, 8.5, W - 2 * M):
        ax.text(M, y, line, color=FAINT, fontsize=8.5, va="top")
        y += 0.17
    height = y + 0.25

    fig.set_size_inches(W, height)
    ax.set_xlim(0, W)
    ax.set_ylim(height, 0)

    png, svg = path.with_suffix(".png"), path.with_suffix(".svg")
    fig.savefig(png, dpi=DPI)
    fig.savefig(svg, format="svg", metadata={"Date": None})
    plt.close(fig)
    return png, svg, height


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
        dict(name="C3", op="5×5 conv, 16 maps, partial (table below)",
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

    fig, ax = new_canvas()
    c5_share = 100 * p["c5"] / total
    y = draw_header(fig, ax, 0.4,
                    f"LeNet-5  ·  LeCun et al., 1998  ·  {total:,} parameters",
                    f"C5 alone holds {c5_share:.1f}% of the parameters. The RBF prototypes "
                    f"are fixed, not trained, so the output layer adds none.",
                    ["conv", "sub", "fc"])
    y = draw_layers(fig, ax, y, rows, total)

    # --- C3 connection grid (Table I): rows = S2 maps, cols = C3 maps ---
    y += 0.35
    ax.plot([M, W - M], [y - 0.15, y - 0.15], color=RULE, lw=0.8)
    ax.text(M, y + 0.05, "C3 connection table (paper, Table I)", color=INK,
            fontsize=12, fontweight="bold", va="top")
    y += 0.55

    grid_x0 = M + 0.55
    cell = (W - M - grid_x0) / 16
    fill = 0.84

    ax.text((grid_x0 + W - M) / 2, y, "C3 feature map", color=INK_2, fontsize=9,
            ha="center", va="top")
    y += 0.22
    for j in range(16):
        ax.text(grid_x0 + (j + 0.5) * cell, y, str(j), color=INK_2, fontsize=8.5,
                family=MONO, ha="center", va="top")
    y += 0.2

    grid_y0 = y
    for j, conn in enumerate(C3_CONNECTIONS):
        for i in range(6):
            ax.add_patch(Rectangle(
                (grid_x0 + (j + (1 - fill) / 2) * cell, grid_y0 + (i + (1 - fill) / 2) * cell),
                fill * cell, fill * cell,
                facecolor=CONV if i in conn else EMPTY, edgecolor="none"))
    for i in range(6):
        ax.text(grid_x0 - 0.08, grid_y0 + (i + 0.5) * cell, str(i), color=INK_2,
                fontsize=8.5, family=MONO, ha="right", va="center")
    ax.text(M + 0.05, grid_y0 + 3 * cell, "S2 feature map", color=INK_2, fontsize=9,
            rotation=90, ha="center", va="center")
    y = grid_y0 + 6 * cell + 0.06

    # inputs per C3 map
    ax.text(grid_x0 - 0.08, y, "inputs", color=FAINT, fontsize=8, ha="right", va="top")
    for j in range(16):
        ax.text(grid_x0 + (j + 0.5) * cell, y, str(len(C3_CONNECTIONS[j])),
                color=INK_2, fontsize=8.5, family=MONO, ha="center", va="top")
    y += 0.3

    # brackets for the four groups
    for a, b, label in [(0, 6, "3 contiguous"), (6, 12, "4 contiguous"),
                        (12, 15, "4 non-\ncontiguous"), (15, 16, "all")]:
        xa, xb = grid_x0 + (a + 0.1) * cell, grid_x0 + (b - 0.1) * cell
        ax.plot([xa, xa, xb, xb], [y, y + 0.07, y + 0.07, y], color=INK_2, lw=0.8)
        ax.text((xa + xb) / 2, y + 0.14, label, color=INK_2, fontsize=8.5,
                ha="center", va="top", linespacing=1.2)
    y += 0.6

    links = sum(len(c) for c in C3_CONNECTIONS)
    summary = (f"{links} of 96 possible S2→C3 links. "
               f"{p['c3']:,} parameters instead of {16 * (6 * 25 + 1):,} for full "
               f"connection. Each S2 map feeds exactly "
               f"{sum(0 in c for c in C3_CONNECTIONS)} C3 maps.")
    for line in wrap(fig, summary, 10, W - 2 * M):
        ax.text(M, y, line, color=INK_2, fontsize=10, va="top")
        y += 0.2
    y += 0.2

    png, svg, height = finish(
        fig, ax, y,
        "Generated by make_diagrams.py from lenet5/lenet5.py. "
        "Counts and shapes are measured from the model, not typed in.", path)
    return total, png, svg, height


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

    fig, ax = new_canvas()
    fc1 = n_params(model, c + "fc1")
    conv_total = sum(r["params"] for r in rows if r["kind"] == "conv")
    y = draw_header(fig, ax, 0.4,
                    f"AlexNet  ·  Krizhevsky et al., 2012  ·  {total:,} parameters",
                    f"fc1 alone holds {100 * fc1 / total:.1f}% of the parameters; all five "
                    f"conv layers together hold {100 * conv_total / total:.1f}%.",
                    ["conv", "fc"])
    y = draw_layers(fig, ax, y, rows, total)

    png, svg, height = finish(
        fig, ax, y + 0.2,
        "Generated by make_diagrams.py from alexnet/alexnet.py. "
        "Counts and shapes are measured from the model, not typed in.", path)
    return total, png, svg, height


if __name__ == "__main__":
    warn_missing_fonts()
    print(f"fonts: labels {SANS}, shapes/counts {MONO}")
    for name, draw in [("lenet5", lenet5_diagram), ("alexnet", alexnet_diagram)]:
        total, png, svg, height = draw(ROOT / name / name)
        print(f"wrote {png.relative_to(ROOT)} + {svg.name}  "
              f"({W:g} × {height:.2f} in, {total:,} parameters)")
    warn_missing_fonts()   # again at the end, so it isn't scrolled away
