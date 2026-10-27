"""
Every figure in this folder, drawn from results/*.json.

Nothing here measures anything. train_cifar.py and floor.py do the measuring and
write their numbers to results/; verify.py checks those files against
themselves. This script only draws them, so every number on every figure is
traceable to a line in a JSON file rather than to something typed into a
plotting script and never checked again.

Each figure is written twice, as <name>.svg and <name>-dark.svg, so a README can
serve the right one to each reader:

    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="figures/name-dark.svg">
      <img src="figures/name.svg">
    </picture>

Three things are checked on every sheet before it is accepted, because each one
has gone wrong here before and none of them raises on its own:

  overlaps     two legends, or a legend and an axis label, drawn on top of each
               other. ax.legend() called twice silently REPLACES the first
               legend, and a legend sized to its longest label grows until it
               covers the curve behind it.
  off-canvas   an artist outside the figure. bbox_inches="tight" used to hide
               this by quietly growing the canvas; it is not used here, because
               it also makes the SVG non-reproducible.
  clipped      data drawn outside its own axis limits. Figure 6 clips on
               purpose -- the paper's own panel stops at 20% while these runs
               start near 78% -- so an axes that means it says so with
               clip_ok(), and the count is printed either way. Any other
               clipping is a silent lie about the data.

    python figures.py                  # all of them
    python figures.py --only figure6   # one
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("RESNET_OUT") or os.path.join(HERE, "results")
FIGURES = os.environ.get("RESNET_FIGS") or os.path.join(HERE, "figures")

# One width for all six, so the README does not step between sizes.
WIDTH_IN = 9.0

PNG_DPI = 192
WRITE_PNG = False
WRITE_WEBP = False
CREATOR = "figures.py"

BODY = 9.0
SMALL = 7.6
TITLE = 10.0

LIGHT = dict(bg="#FBF9F4", ink="#0F172A", muted="#5B6762", rule="#D9D3C4",
             accent="#371E77", warn="#D3542A", good="#4F7357", fill="#EFEADF")
DARK = dict(bg="#0D1117", ink="#E6EDF3", muted="#8B949E", rule="#30363D",
            accent="#A78BFA", warn="#FF7B5A", good="#7EE2A8", fill="#161B22")

# Depth alone cannot separate plain20 from resnet20 -- they share a depth. Where
# both families sit on one axis, colour carries the MODEL: warm = plain,
# cool = residual. On the light palette the ramp darkens with depth; on the dark
# palette it brightens instead, because a near-black #7D2B2B is invisible on
# #0D1117. Same ordering, opposite direction, so "deeper is further along the
# ramp" reads the same way in both.
MODEL = {
    id(LIGHT): {"plain20": "#E8903C", "plain32": "#D9534F", "plain44": "#A23B3B",
                "plain56": "#7D2B2B", "resnet20": "#5BA7DB", "resnet32": "#3B7FB5",
                "resnet44": "#2A6094", "resnet56": "#17365D"},
    id(DARK): {"plain20": "#A85A1C", "plain32": "#D06B42", "plain44": "#EC8D6C",
               "plain56": "#FFAE90", "resnet20": "#1A6AAE", "resnet32": "#3F8FCB",
               "resnet44": "#67B0E2", "resnet56": "#8CCBF0"},
}
# Inside a single-family panel, depth is unambiguous and the four depths want to
# be told apart at a glance rather than ordered.
DEPTH = {
    id(LIGHT): {20: "#D62728", 32: "#E07B39", 44: "#7D3C98", 56: "#1F77B4",
                110: "#117A65"},
    id(DARK): {20: "#FF7B6B", 32: "#FFB066", 44: "#C39BE8", 56: "#6BB6E8",
               110: "#7EE2A8"},
}
KIND = {
    id(LIGHT): {"plain": "#D62728", "resnet": "#1F77B4"},
    id(DARK): {"plain": "#FF7B6B", "resnet": "#6BB6E8"},
}
SCHED_MARKER = {"paper": "o", "tail3x": "s", "long3x": "D", "short": "^", "smoke": "v"}

TRAIN = dict(lw=1.7, ls="-")                  # solid  = training error
TEST = dict(lw=0.95, ls="--", alpha=0.8)      # dashed = test error

CONVENTION = ("solid = training error, dashed = test error; vertical lines are the "
              "learning-rate drops")
REBUILT_NOTE = ("hollow markers: runs rebuilt from notebook output after a lost "
                "runtime -- same measurements, two decimals instead of three")

TABLE_6_TEST = {20: 8.75, 32: 7.51, 44: 7.17, 56: 6.97, 110: 6.43}

FAMILY = "Inter"

# The family name alone does not pin the drawing: Inter 4.000 and 4.001 differ
# in outlines and spacing, so the same script under a neighbouring build moves
# glyphs and rewrites paths. 4.001 is the reference. A mismatch warns rather
# than refuses -- the figure is still usable, and the three checks still have to
# pass -- and every SVG records the version that drew it.
FONT_VERSION = "4.001"

# Every panel size, legend placement and note line here was laid out against
# Inter. DejaVu Sans, matplotlib's default, is wide enough to push legends over
# the curves behind them. That used to happen silently, through a fallback; it
# now stops instead. --font overrides it for anyone who would rather have a
# figure in the wrong face than none, and the checks still apply to whatever
# comes out.
#
# The SVGs need no font installed to READ: svg.fonttype is "path", so the text
# ships as outlines. The font is a requirement for generating, not for viewing.
FONT_HELP = """
Install Inter (not Inter Tight, which is a different family with different
metrics):

    macOS     brew install --cask font-inter
    Linux     apt install fonts-inter   (or unzip rsms/inter into ~/.fonts)
    any       https://rsms.me/inter/  then
              python -c "import matplotlib.font_manager as f; f._load_fontmanager(try_read_cache=False)"

or re-run with --font "<installed family>" to accept a different face.
"""


def font_version(family):
    """The version out of the font file's own name table, or None.

    Read from the file rather than from a package manager, so it reports the
    face matplotlib will actually draw with.
    """
    import matplotlib.font_manager as fm
    from matplotlib import ft2font
    try:
        table = ft2font.FT2Font(fm.findfont(fm.FontProperties(family=family))).get_sfnt()
    except Exception:
        return None
    for key, value in table.items():
        if key[3] == 5:                                   # name id 5: version
            text = value.decode("utf-16-be", "ignore") if b"\x00" in value \
                else value.decode("utf8", "ignore")
            return text.replace("Version", "").strip()
    return None


def resolve_font(requested):
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    if requested in have:
        return requested
    near = sorted(n for n in have if requested.split()[0].lower() in n.lower())
    raise SystemExit(f"figures.py: the font {requested!r} is not installed, and these "
                     f"figures are laid out for it.\n"
                     + (f"Similar families found: {', '.join(near[:6])}\n" if near else "")
                     + FONT_HELP)


def write_webp(png_path, keep_png):
    """Lossless WebP beside the PNG, from the same pixels.

    Lossless, not quality 95: these are flat colour and thin lines, where lossy
    compression puts halos on hairlines and saves nothing worth having.
    """
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("figures.py: --webp needs Pillow (pip install Pillow), "
                         "or use --png and convert the files yourself.")
    out = png_path[:-4] + ".webp"
    Image.open(png_path).convert("RGB").save(out, "WEBP", lossless=True, method=6)
    if not keep_png:
        os.remove(png_path)


# ------------------------------------------------------------------ the runs
NAME = re.compile(r"^(plain\d+|resnet\d+)_(\w+?)(?:_seed(\d+))?$")
RUNS = {}
USED = defaultdict(set)          # figure name -> the run keys it drew


def load_runs():
    for path in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        m = NAME.match(os.path.splitext(os.path.basename(path))[0])
        if not m:
            continue
        with open(path) as f:
            RUNS[(m.group(1), m.group(2), int(m.group(3) or 0))] = json.load(f)
    if not RUNS:
        raise SystemExit(
            f"figures.py: no result JSONs in {RESULTS}\n"
            f"Expected <model>_<schedule>[_seed<n>].json. Set RESNET_OUT to point\n"
            f"somewhere else, or run train_cifar.py first.")
    return RUNS


def rebuilt(key):
    return bool(RUNS[key]["_provenance"].get("RECONSTRUCTED"))


def seeds_of(model, sched):
    return sorted(s for (m, c, s) in RUNS if m == model and c == sched)


def keys_of(model, sched):
    return [(model, sched, s) for s in seeds_of(model, sched)]


def finals(model, sched, field):
    return [RUNS[k][field] for k in keys_of(model, sched)]


def minutes(key):
    return (RUNS[key]["_provenance"].get("seconds") or 0) / 60


def pick(model, sched):
    """One representative run: the lowest seed present."""
    ks = keys_of(model, sched)
    return ks[0] if ks else None


def present(kind, sched="paper"):
    return [d for d in (20, 32, 44, 56, 110) if seeds_of(f"{kind}{d}", sched)]


def schedules():
    order = ("paper", "tail3x", "long3x", "short", "smoke")
    found = {c for (_, c, _) in RUNS}
    return [s for s in order if s in found] + sorted(found - set(order))


# ------------------------------------------------------------------ the canvas
class Canvas:
    """A palette-aware grid of axes that refuses to save a broken sheet."""

    def __init__(self, pal, nrows=1, ncols=1, height=4.0, **kw):
        self.p = pal
        self.fig, self.axs = plt.subplots(nrows, ncols, squeeze=False,
                                          figsize=(WIDTH_IN, height), **kw)
        self.fig.patch.set_facecolor(pal["bg"])
        self._clip_ok = set()
        for ax in self.flat:
            self.style(ax)

    @property
    def flat(self):
        return list(self.axs.flat)

    def style(self, ax):
        p = self.p
        ax.set_facecolor(p["bg"])
        ax.set_axisbelow(True)
        ax.grid(True, color=p["rule"], lw=0.7)
        for side in ax.spines.values():
            side.set_color(p["rule"])
            side.set_linewidth(0.9)
        ax.tick_params(colors=p["muted"], labelsize=SMALL, length=3)
        ax.xaxis.label.set_color(p["ink"])
        ax.yaxis.label.set_color(p["ink"])
        ax.title.set_color(p["ink"])
        ax.title.set_fontsize(TITLE)

    def clip_ok(self, ax):
        """Declare that this axes clips its data deliberately."""
        self._clip_ok.add(id(ax))

    def lr_marks(self, ax, key):
        """Darker than the grid on purpose: on a 192k-wide panel the drops sit
        close together near the left edge and would read as grid lines.

        Labelled with a leading underscore so clipped() skips them -- axvline
        spans the axes in y, which is not data going off the top.
        """
        for drop in RUNS[key]["_provenance"]["schedule"]["lr_drops"]:
            ax.axvline(drop, color=self.p["muted"], lw=0.9, alpha=0.65,
                       zorder=0, label="_lrdrop")

    def iter_axis(self, ax):
        """192,000 in full digits does not fit five times across a narrow panel."""
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))

    def face(self, key, colour):
        """Rebuilt runs are hollow, so they are never silently mixed in."""
        return self.p["bg"] if rebuilt(key) else colour

    def legend(self, ax, handles=None, **kw):
        kw.setdefault("frameon", False)
        kw.setdefault("fontsize", SMALL)
        lg = ax.legend(**kw) if handles is None else ax.legend(handles=handles, **kw)
        for t in lg.get_texts():
            t.set_color(self.p["ink"])
        return lg

    def style_key(self, ax, loc="lower left"):
        """The train/test key, kept as a SECOND legend.

        ax.legend() called twice replaces the first legend rather than adding to
        it, which is how the model legend went missing once already. add_artist
        is what keeps both.
        """
        first = ax.get_legend()
        if first is not None:
            ax.add_artist(first)
        return self.legend(
            ax, loc=loc, handles=[
                Line2D([], [], color=self.p["muted"], **TRAIN, label="train"),
                Line2D([], [], color=self.p["muted"], **TEST, label="test")])

    def suptitle(self, text):
        self.fig.suptitle(text, fontsize=BODY, color=self.p["ink"])

    def finish(self, bottom=None):
        """tight_layout, and nothing else.

        tight_layout already measures the suptitle and the panel titles and
        leaves room for both. Passing it a rect as well reserves the top band
        twice over, which is where the acre of white between the title and the
        panels came from; setting `top` by hand afterwards instead pushes the
        panel titles up into the suptitle. Both were tried, and the overlap
        check caught both. This is the version that holds for a title of any
        number of lines.
        """
        self.fig.tight_layout()
        if bottom is not None:
            self.fig.subplots_adjust(bottom=bottom)

    # -------------------------------------------------------------- the checks
    def _boxes(self, r):
        out = []
        for i, ax in enumerate(self.flat):
            for lg in ax.findobj(Legend):
                label = lg.get_texts()[0].get_text() if lg.get_texts() else "?"
                out.append((f"panel {i} legend ({label[:26]})", lg.get_window_extent(r)))
            for name, artist in (("title", ax.title), ("xlabel", ax.xaxis.label),
                                 ("ylabel", ax.yaxis.label)):
                if artist.get_text().strip():
                    out.append((f"panel {i} {name}", artist.get_window_extent(r)))
        for lg in self.fig.legends:
            label = lg.get_texts()[0].get_text() if lg.get_texts() else "?"
            out.append((f"figure legend ({label[:26]})", lg.get_window_extent(r)))
        if self.fig._suptitle is not None and self.fig._suptitle.get_text().strip():
            out.append(("suptitle", self.fig._suptitle.get_window_extent(r)))
        return out

    def overlaps(self):
        self.fig.canvas.draw()
        boxes = self._boxes(self.fig.canvas.get_renderer())
        hits = []
        for i, (n1, b1) in enumerate(boxes):
            for n2, b2 in boxes[i + 1:]:
                ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if ix > 1.0 and iy > 1.0:
                    hits.append((n1, n2))
        return hits

    def offcanvas(self):
        self.fig.canvas.draw()
        w, h = self.fig.get_size_inches() * self.fig.dpi
        out = []
        for name, b in self._boxes(self.fig.canvas.get_renderer()):
            if b.x0 < -1 or b.y0 < -1 or b.x1 > w + 1 or b.y1 > h + 1:
                out.append((name, b.x0, b.y0, b.x1, b.y1, w, h))
        return out

    def clipped(self):
        """(panel, points outside the limits, points drawn, declared) per axes."""
        report = []
        for i, ax in enumerate(self.flat):
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            pts = []
            for line in ax.lines:
                if str(line.get_label()).startswith("_"):
                    continue
                pts += list(zip(line.get_xdata(), line.get_ydata()))
            # Only scatter. A LineCollection -- which is what errorbar's
            # whiskers are -- has no per-point offsets, and get_offsets() hands
            # back the default (0, 0) rather than nothing. On a log axis that
            # phantom point reads as data off the bottom of the panel, and a
            # check that cries wolf is a check that gets switched off.
            for col in ax.collections:
                if isinstance(col, mcoll.PathCollection):
                    pts += [tuple(o) for o in col.get_offsets()]
            bad = sum(1 for x, y in pts
                      if not (x0 - 1e-9 <= x <= x1 + 1e-9
                              and y0 - 1e-9 <= y <= y1 + 1e-9))
            if bad:
                report.append((i, bad, len(pts), id(ax) in self._clip_ok))
        return report

    def save(self, name):
        hits = self.overlaps()
        off = self.offcanvas()
        clip = self.clipped()
        suffix = "-dark" if self.p is DARK else ""
        os.makedirs(FIGURES, exist_ok=True)
        path = os.path.join(FIGURES, f"{name}{suffix}.svg")
        # metadata Date=None drops the timestamp and svg.hashsalt (set in main)
        # fixes the element ids, so regenerating an unchanged figure produces an
        # unchanged file and "does this still reproduce?" can be answered by
        # comparing bytes. bbox_inches is deliberately not passed: it rescales
        # the canvas, which breaks that and hides anything off-canvas.
        self.fig.savefig(path, format="svg", facecolor=self.p["bg"],
                         metadata={"Date": None, "Creator": CREATOR})
        if (WRITE_PNG or WRITE_WEBP) and self.p is LIGHT:
            png = os.path.join(FIGURES, "png", f"{name}.png")
            self.fig.savefig(png, format="png", dpi=PNG_DPI, facecolor=self.p["bg"])
            if WRITE_WEBP:
                write_webp(png, keep_png=WRITE_PNG)
        plt.close(self.fig)

        for n1, n2 in hits:
            print(f"   !! {name}{suffix}: overlap — {n1} / {n2}")
        for n, x0, y0, x1, y1, w, h in off:
            print(f"   !! {name}{suffix}: {n} is off-canvas "
                  f"({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) of {w:.0f}x{h:.0f}")
        undeclared = False
        for i, bad, total, declared in clip:
            if declared:
                print(f"      {name}{suffix}: panel {i} clips {bad} of {total} "
                      f"points (declared)")
            else:
                print(f"   !! {name}{suffix}: panel {i} draws {bad} of {total} points "
                      f"outside its own limits, and did not declare it")
                undeclared = True
        return not hits and not off and not undeclared


# ===================================================== 1. Figure 6 reproduced
def fig_figure6(pal):
    c = Canvas(pal, 1, 2, height=4.1, sharey=True)
    for ax, (title, kind) in zip(c.flat, [("plain nets", "plain"),
                                          ("residual nets", "resnet")]):
        first = None
        for d in present(kind):
            key = pick(f"{kind}{d}", "paper")
            USED["figure6"].add(key)
            first = first or key
            h = RUNS[key]["history"]
            it = [p["iteration"] for p in h]
            colour = DEPTH[id(pal)][d]
            ax.plot(it, [p["train_error"] for p in h], color=colour,
                    label=f"{kind}{d}", **TRAIN)
            ax.plot(it, [p["test_error"] for p in h], color=colour, **TEST)
        if first:
            c.lr_marks(ax, first)
        ax.set_title(title)
        ax.set_xlabel("iteration")
        c.iter_axis(ax)
        # The paper's own panel stops at 20%. These runs start near 78%, so the
        # first few thousand iterations are off the top of the axes by design.
        ax.set_ylim(0, 20)
        ax.set_yticks(np.arange(0, 20.1, 2.5))
        c.clip_ok(ax)
        c.legend(ax, loc="upper right")
        c.style_key(ax)
    c.flat[0].set_ylabel("error (%)")
    c.suptitle("Figure 6, reproduced — CIFAR-10, the paper's schedule, one seed "
               f"per depth\n{CONVENTION}")
    c.finish()
    return c.save("figure6")


# ===================================================== 2. every seed, separately
def fig_seeds(pal):
    c = Canvas(pal, 2, 2, height=6.4, sharex=True)
    for col, kind in enumerate(("plain", "resnet")):
        for row, metric in enumerate(("train_error", "test_error")):
            ax = c.axs[row][col]
            first = None
            for d in present(kind):
                ks = keys_of(f"{kind}{d}", "paper")
                USED["seeds"].update(ks)
                first = first or ks[0]
                colour = DEPTH[id(pal)][d]
                field = "final_train_error" if row == 0 else "final_test_error"
                fin = finals(f"{kind}{d}", "paper", field)
                for j, key in enumerate(ks):
                    h = RUNS[key]["history"]
                    ax.plot([p["iteration"] for p in h], [p[metric] for p in h],
                            color=colour, lw=1.1, alpha=0.8,
                            label=(f"{kind}{d} · {len(ks)} seeds · "
                                   f"{min(fin):.2f}–{max(fin):.2f}%"
                                   if len(ks) > 1 else
                                   f"{kind}{d} · 1 seed · {fin[0]:.2f}%")
                            if j == 0 else None)
                    # The clean final pass is a different quantity from the
                    # running curve above it, so it is a marker and is never
                    # joined to the line.
                    ax.scatter([h[-1]["iteration"]], [RUNS[key][field]],
                               facecolor=c.face(key, colour), edgecolor=colour,
                               lw=1.2, s=26, zorder=4)
            if first:
                c.lr_marks(ax, first)
            ax.set_yscale("log")
            ax.set_ylabel(("training" if row == 0 else "test") + " error (%), log")
            ax.set_title(f"{kind} nets — " + ("training" if row == 0 else "test"))
            c.legend(ax, loc="upper right")
            c.iter_axis(ax)
            if row == 1:
                ax.set_xlabel("iteration")
    c.suptitle("Every seed drawn separately; the width of a bundle is the "
               f"run-to-run noise\n{REBUILT_NOTE}")
    c.finish()
    return c.save("seeds")


# ===================================================== 3. error against depth
def fig_depth(pal):
    c = Canvas(pal, 1, 2, height=3.9)
    for ax, field, title in ((c.flat[0], "final_train_error", "final training error"),
                             (c.flat[1], "final_test_error", "final test error")):
        for kind in ("plain", "resnet"):
            colour = KIND[id(pal)][kind]
            xs, ys, lo, hi = [], [], [], []
            for d in present(kind):
                ks = keys_of(f"{kind}{d}", "paper")
                USED["depth"].update(ks)
                vals = [RUNS[k][field] for k in ks]
                mean = float(np.mean(vals))
                xs.append(d)
                ys.append(mean)
                lo.append(mean - min(vals))
                hi.append(max(vals) - mean)
                for k in ks:
                    ax.scatter([d], [RUNS[k][field]], facecolor=c.face(k, colour),
                               edgecolor=colour, lw=1.1, s=20, zorder=4)
            if xs:
                ax.errorbar(xs, ys, yerr=[lo, hi], color=colour, lw=1.6, capsize=3,
                            marker="None", label=f"{kind} nets, seed mean")
        ax.set_xlabel("depth (weight layers)")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title)
        ax.set_xticks([20, 32, 44, 56])
        ax.margins(x=0.14, y=0.16)
        c.legend(ax, loc="upper left")
    c.flat[0].set_yscale("log")
    c.flat[0].set_title("final training error (log)")
    c.suptitle("Line = seed mean, whisker = full range, dot = one run\n"
               "the 20→32 step is inside plain56's own spread, and plain32 and "
               "plain44 are one seed each")
    c.finish()
    return c.save("depth")


# ===================================================== 4. the three schedules
def fig_schedules(pal):
    scheds = [s for s in schedules() if any(pick(f"plain{d}", s) for d in (20, 56))]
    c = Canvas(pal, 1, len(scheds), height=4.3, sharey=True)
    for ax, sched in zip(c.flat, scheds):
        first = None
        for d in (20, 56):
            key = pick(f"plain{d}", sched)
            if key is None:
                continue
            USED["schedules"].add(key)
            first = first or key
            colour = DEPTH[id(pal)][d]
            h = RUNS[key]["history"]
            it = [p["iteration"] for p in h]

            # The paper schedule's own seed range, as a band. Without it the eye
            # compares one draw against one draw, and plain56 on the paper
            # schedule spans more than two points between seeds.
            band = finals(f"plain{d}", "paper", "final_train_error")
            shaded = len(band) > 1 and sched != "paper"
            if shaded:
                ax.axhspan(min(band), max(band), color=colour, alpha=0.11, zorder=0)
            label = f"plain{d} · {RUNS[key]['final_train_error']:.2f}% train"
            if shaded:
                label += f"\n   (paper {min(band):.2f}–{max(band):.2f})"
            ax.plot(it, [p["train_error"] for p in h], color=colour, **TRAIN,
                    label=label)
            ax.plot(it, [p["test_error"] for p in h], color=colour, **TEST)
            ax.scatter([it[-1]], [RUNS[key]["final_train_error"]],
                       facecolor=c.face(key, colour), edgecolor=colour,
                       lw=1.2, s=26, zorder=4)
        if first:
            c.lr_marks(ax, first)
            sc = RUNS[first]["_provenance"]["schedule"]
            ax.set_title(f"{sched} · {sc['iterations'] // 1000}k iters · drops "
                         + "/".join(f"{v // 1000}k" for v in sc["lr_drops"]))
        ax.set_xlabel("iteration")
        c.iter_axis(ax)
        ax.set_ylim(0, 20)
        ax.set_yticks(np.arange(0, 20.1, 2.5))
        c.clip_ok(ax)
        c.legend(ax, loc="upper right")
    c.style_key(c.flat[0])
    c.flat[0].set_ylabel("error (%)")
    # The band and the dot are both FINAL clean errors; the curve is the running
    # error, averaged over a window that scales with the schedule. Saying so on
    # the sheet, because a reader comparing the curve's end against the band is
    # comparing two different quantities.
    c.suptitle("What a longer schedule buys the plain nets\nband = final-error "
               "range over seeds on the paper schedule, dot = this run's final "
               "error; the curves are running error")
    c.finish()
    return c.save("schedules")


# ===================================================== 5. against Table 6
def fig_table6(pal):
    c = Canvas(pal, 1, 1, height=3.9)
    ax = c.flat[0]
    cool, warm = KIND[id(pal)]["resnet"], KIND[id(pal)]["plain"]
    depths = present("resnet")
    for i, d in enumerate(depths):
        ks = keys_of(f"resnet{d}", "paper")
        USED["table6"].update(ks)
        vals = [RUNS[k]["final_test_error"] for k in ks]
        for j, k in enumerate(ks):
            ax.scatter([d], [RUNS[k]["final_test_error"]], facecolor=c.face(k, cool),
                       edgecolor=cool, lw=1.3, s=38, zorder=3,
                       label="this reproduction" if i == 0 and j == 0 else None)
        if len(vals) > 1:
            ax.plot([d, d], [min(vals), max(vals)], color=cool, lw=1.4, alpha=0.6)
            ax.annotate(f"{min(vals):.2f}–{max(vals):.2f}", (d, min(vals)),
                        textcoords="offset points", xytext=(12, -3),
                        fontsize=SMALL, color=cool)
        if d in TABLE_6_TEST:
            ax.scatter([d], [TABLE_6_TEST[d]], marker="_", s=620, color=warm,
                       lw=2.2, zorder=4, label="Table 6" if i == 0 else None)
            ax.annotate(f"{TABLE_6_TEST[d]:.2f}", (d, TABLE_6_TEST[d]),
                        textcoords="offset points", xytext=(12, 4),
                        fontsize=SMALL, color=warm)
    ax.set_xlabel("depth (weight layers)")
    ax.set_ylabel("test error (%)")
    ax.set_xticks(depths)
    ax.margins(x=0.22, y=0.22)
    c.legend(ax, loc="upper right")
    c.suptitle("Every residual run against the published figure\n6.97 sits inside "
               "the four-seed range at depth 56 — there is no systematic offset "
               "to explain")
    c.finish()
    return c.save("table6")


# ===================================================== 6. what the time bought
def fig_cost(pal):
    # No per-point text: with 22 runs the labels collide with each other and
    # with any legend inside the axes. Colour carries the model, marker carries
    # the schedule, and both keys sit under the panels where nothing overlaps.
    models_seen = sorted({m for (m, _, _) in RUNS},
                         key=lambda m: (m.startswith("resnet"),
                                        int(re.sub(r"\D", "", m))))
    scheds = schedules()
    c = Canvas(pal, 1, 2, height=4.6)
    for ax, field, title in ((c.flat[0], "final_train_error", "final training error"),
                             (c.flat[1], "final_test_error", "final test error")):
        for k in sorted(RUNS):
            model, sched, _ = k
            USED["cost"].add(k)
            colour = MODEL[id(pal)].get(model, pal["muted"])
            ax.scatter([minutes(k)], [RUNS[k][field]],
                       marker=SCHED_MARKER.get(sched, "P"),
                       facecolor=c.face(k, colour), edgecolor=colour,
                       lw=1.3, s=46, zorder=3)
        ax.set_xlabel("wall clock (minutes on one A100)")
        ax.set_ylabel(f"{title} (%)")
        ax.set_title(title)
        ax.margins(0.13)
    c.flat[0].set_yscale("log")
    c.flat[0].set_title("final training error (log)")

    c.suptitle("Error against the compute that bought it — colour = model, marker "
               "= schedule\nthe figure behind the claim that the shortcut beats 3x "
               "the iterations")
    c.finish(bottom=0.215)

    def key(handles, y):
        c.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y),
                     ncol=len(handles), frameon=False, fontsize=SMALL,
                     labelcolor=pal["ink"], handletextpad=0.4, columnspacing=1.2)

    key([Line2D([], [], ls="none", marker="o", markersize=5,
                color=MODEL[id(pal)].get(m, pal["muted"]), label=m)
         for m in models_seen], 0.065)
    key([Line2D([], [], ls="none", marker=SCHED_MARKER.get(s, "P"), markersize=5,
                color=pal["muted"], label=s) for s in scheds], 0.008)
    return c.save("cost")


FIGURES_ALL = {"figure6": fig_figure6, "seeds": fig_seeds, "depth": fig_depth,
               "schedules": fig_schedules, "table6": fig_table6, "cost": fig_cost}


def coverage():
    """Which figure drew which run, and which runs no figure drew.

    A result file that no figure touches sits in results/ looking like evidence
    for something. This is the only thing that notices.
    """
    print("\ncoverage — which figure drew which run")
    drawn = set().union(*USED.values()) if USED else set()
    for k in sorted(RUNS):
        where = sorted(f for f, ks in USED.items() if k in ks)
        mark = "  " if where else "  NOT DRAWN ->"
        print(f"{mark} {k[0]:<9} {k[1]:<7} seed {k[2]}   {', '.join(where) or '(none)'}")
    missing = sorted(set(RUNS) - drawn)
    print(f"\n{len(drawn)} of {len(RUNS)} runs appear in at least one figure")
    if missing:
        print("THESE RESULTS ARE IN results/ BUT ON NO FIGURE:")
        for k in missing:
            print(f"  {k[0]} {k[1]} seed {k[2]}")
    else:
        print("every result file in results/ is on at least one figure")
    return not missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES_ALL))
    ap.add_argument("--png", action="store_true",
                    help="also write figures/png/<name>.png at %d dpi, light palette "
                         "only, for somewhere that cannot serve SVG" % PNG_DPI)
    ap.add_argument("--webp", action="store_true",
                    help="write figures/png/<name>.webp instead, lossless; "
                         "--png --webp keeps both")
    ap.add_argument("--font", default=FAMILY,
                    help=f"font family to lay the figures out in (default {FAMILY}); "
                         "anything else re-flows every measured position")
    args = ap.parse_args()

    family = resolve_font(args.font)
    version = font_version(family)

    global CREATOR, WRITE_PNG, WRITE_WEBP
    CREATOR = f"figures.py — {family} {version or 'version unknown'}"
    print(f"   font: {family} {version or '(version unreadable)'}")
    if family == FAMILY and version and not version.startswith(FONT_VERSION):
        print(f"   !! this is not the build these figures were measured against "
              f"({FONT_VERSION}). The checks below still apply, but the committed "
              f"SVGs will not reproduce byte for byte.")
    WRITE_PNG, WRITE_WEBP = args.png, args.webp

    load_runs()
    print(f"   {len(RUNS)} runs from {RESULTS}")

    os.makedirs(FIGURES, exist_ok=True)
    if args.png or args.webp:
        os.makedirs(os.path.join(FIGURES, "png"), exist_ok=True)
    plt.rcParams.update({"font.family": family, "font.size": BODY,
                         "svg.fonttype": "path",       # text as paths: no font to read
                         "svg.hashsalt": "resnet"})    # deterministic element ids

    ok = True
    for name in (args.only or list(FIGURES_ALL)):
        for pal in (LIGHT, DARK):
            ok &= FIGURES_ALL[name](pal)
        print(f"   {name}.svg + {name}-dark.svg")

    # Coverage only means anything when every figure ran.
    if not args.only:
        ok &= coverage()
    print("\nall figures pass their checks" if ok else "\nSOME FIGURE FAILED ITS CHECKS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
