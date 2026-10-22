"""
Build VGG.ipynb — the study as a narrative rather than as a scratch log.

The notebook that was in this folder first was the real session: 44 code cells
in the order things were tried, no headings, no prose, no figures. Useful to the
person who wrote it and to nobody else.

This builds the replacement. Cells that need only PyTorch actually run and their
output is captured, so the notebook reads end to end on GitHub without anyone
cloning it. Cells that need an A100 or ImageNet are marked and read their numbers
back from results/ instead of pretending to compute them.

    python build_notebook.py          # build and execute
    python build_notebook.py --no-exec
"""

import argparse
import json
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))


CELLS = [
md("""
# VGG, read by running it

[*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, ICLR 2015.

This notebook works through the paper in the order the questions actually came up: build the six
architectures and check them against the paper's own tables, ask what the tables leave out, and then
re-run the paper's test-time procedure on the paper's own validation set.

**What this is not.** Nothing here is trained to convergence on ImageNet. The architectures are built
from scratch and verified; the evaluation study uses torchvision's pretrained configuration D. Where a
claim could not be reproduced, that is said rather than worked around — there is a list at the bottom.

Cells marked **⏱ needs a GPU** or **⏱ needs ImageNet** read their numbers back from `results/`. Every
other cell runs here.
"""),

md("""
---
## 1. The six configurations

Table 1's whole design is that depth is the only thing that moves. One receptive field (3×3), one
stride, one padding rule, five max-pool stages, three fully-connected layers. Hold all of that constant,
vary depth, and the error difference can be attributed to depth.

Each configuration is a list read left to right: `(k, channels)` is a convolution, `"M"` a 2×2 max-pool,
`"L"` the local response normalization that only A-LRN has.
"""),

code("""
import torch, torch.nn as nn
from vgg import CONFIGS, WEIGHT_LAYERS, TABLE_2, make_vgg

for name, cfg in CONFIGS.items():
    convs = [x for x in cfg if x != "M" and x != "L"]
    print(f"{name:<6} {len(convs):>2} convolutions + 3 FC = {WEIGHT_LAYERS[name]:>2} weight layers")
"""),

md("""
Padding is `k // 2` on every convolution, which preserves the spatial size for any odd kernel — so the
3×3 layers get padding 1 and configuration C's 1×1 layers get padding 0 from the same rule, with no
special case. Only the five pools change the resolution, which is why 224 → 7 is exactly 2⁵.
"""),

code("""
for name in CONFIGS:
    m = make_vgg(name)
    total = sum(p.numel() for p in m.parameters())
    ok = total == TABLE_2[name]
    print(f"{name:<6} {total:>12,}   Table 2: {TABLE_2[name]:>12,}   {'exact' if ok else 'MISMATCH'}")
"""),

md("""
All six match. The paper prints them rounded to millions, which is why A, A-LRN and B all read 133 there.

A and A-LRN are identical because local response normalization has no parameters. That is correct, and
it is also the first hint that Table 2 is measuring cost in the wrong currency — we come back to it in §3.

![](figures/configs.svg)
"""),

md("""
---
## 2. The 3×3 claim, and the word the cost argument leaves out

§2.3 argues that three stacked 3×3 layers see the same 7×7 receptive field as one 7×7 layer while
costing 27C² weights against 49C² — the single layer is 81% more expensive.

The cost arithmetic is easy to check. The part worth checking is the other half of the claim.
"""),

code("""
C = 64
print(f"three 3x3 over {C} channels: 3 * 3*3*{C}*{C} = {3*3*3*C*C:,}  (27C^2)")
print(f"one   7x7 over {C} channels:     7*7*{C}*{C} = {7*7*C*C:,}  (49C^2)")
print(f"the single layer is {49/27 - 1:.0%} more expensive")
"""),

md("""
**Convolution is linear.** So two stacked 3×3 convolutions with *no activation between them* are not a
deeper model at all — they collapse exactly into one 5×5 convolution. The depth argument in §2.3 is an
argument about the ReLUs, not about the kernels, and the cost comparison never mentions them.

The cleanest way to see it is the impulse response: feed a single bright pixel through the stack and
look at what comes out.
"""),

code("""
torch.manual_seed(0)
a = nn.Conv2d(1, 1, 3, padding=1, bias=False)
b = nn.Conv2d(1, 1, 3, padding=1, bias=False)

impulse = torch.zeros(1, 1, 9, 9); impulse[0, 0, 4, 4] = 1.0
with torch.no_grad():
    stacked = b(a(impulse))            # no activation between them

# the response is 5x5 and nonzero nowhere else: the pair IS a single 5x5 kernel
nz = (stacked.abs() > 1e-7)[0, 0]
rows, cols = nz.any(1).nonzero().flatten(), nz.any(0).nonzero().flatten()
print(f"impulse response occupies rows {rows.min().item()}-{rows.max().item()}, "
      f"cols {cols.min().item()}-{cols.max().item()}  ->  a {rows.max()-rows.min()+1} x "
      f"{cols.max()-cols.min()+1} kernel")

equivalent = nn.Conv2d(1, 1, 5, padding=2, bias=False)
with torch.no_grad():
    equivalent.weight.copy_(stacked[:, :, 2:7, 2:7].flip(-1).flip(-2))
    same = torch.allclose(equivalent(impulse), stacked, atol=1e-6)
print("one 5x5 layer reproduces the pair exactly:", same)
"""),

md("""
Two layers in, one layer out, same function. Put a ReLU between them and no single convolution can
reproduce it — that is the entire content of "deeper".
"""),

md("""
---
## 3. What Table 2 cannot see

**⏱ needs a GPU.** Numbers below are from one NVIDIA A100-SXM4-80GB, batch 32, forward + backward,
median of 25 steps after warmup. Reproduce with `python cost.py`.

Table 2 quantifies cost as parameters. A and A-LRN have the same parameter count and the same
arithmetic, and the paper's only remark on what LRN costs is that it "leads to increased memory
consumption and computation time" (§4.1), with no figure attached.
"""),

code("""
import json
cost = json.load(open("results/cost.json"))
lrn = cost["lrn"]

print(f"{'':<14}{'A':>16}{'A-LRN':>16}{'change':>14}")
for label, key, fmt in [("parameters", "params", "{:,}"), ("mult-adds", "gmacs", "{} G"),
                        ("step time", "step_ms", "{} ms"), ("peak memory", "peak_mb", "{:,} MB")]:
    a, b = lrn["A"][key], lrn["A_LRN"][key]
    d = "none" if a == b else ("+%.1f%%" % lrn["delta"]["step_ms_pct"] if key == "step_ms"
                               else "+{:,} MB".format(lrn["delta"]["peak_mb"]))
    print(f"{label:<14}{fmt.format(a):>16}{fmt.format(b):>16}{d:>14}")

w = lrn["why"]
print(f"\\nPyTorch expands LocalResponseNorm into {w['autograd_ops_in_pytorch_lrn']} autograd operations.")
print(f"One full-size tensor at this batch is {w['one_full_size_tensor_mb']} MB; "
      f"the increase is {w['ratio']}x that.")
"""),

md("""
The first two rows are what Table 2 reports. The last two are what it cannot.

The authors used Caffe, which had a fused LRN kernel — so this cost is this framework's, not theirs,
and there is no honest way to project it backwards onto their machines. What it does establish is that
a layer with no parameters and no arithmetic is not free, and a table of parameter counts will never say so.

### A lesson about which number you measure

`nn.ReLU(inplace=True)` should save memory: VGG-16 at batch 32 stores a lot of activations. Predicted
saving ≈ 784 MB.
"""),

code("""
ip = cost["inplace_relu"]
print(f"predicted saving        {ip['predicted_saving_mb']:>8} MB")
print(f"measured, global peak   {ip['measured']['saving_mb']:>8} MB")
print(f"measured, forward peak  {ip['forward_peak']['saving_mb']:>8} MB")
print("\\n" + ip["reading"])
"""),

md("""
Both measurements are right. The saving is real and it is in the forward pass; it is invisible in the
global peak because that peak is set during *backward*, when the stored activations are being consumed
anyway. Report only the global peak and you report zero for an effect of several hundred megabytes.

### And a second effect: a multiply-accumulate is not a multiply-accumulate
"""),

code("""
print(f"{'step':<9}{'+GMACs':>9}{'ms/GMAC':>10}   what changed")
for s in cost["arithmetic_intensity"]["steps"]:
    print(f"{s['step']:<9}{s['gmacs']:>9.2f}{s['ms_per_gmac']:>10.2f}   {s['changed']}")
"""),

md("""
C→D adds **exactly the same** 3.70 GMACs as A→B and costs 2.5× less per GMAC, because it is added late
on small maps rather than early on large ones. And C's 1×1 convolutions are the worst value in the
table: a 3×3 kernel does nine multiply-accumulates for every weight it loads, a 1×1 does one.

GMAC count alone does not predict time.

![](figures/cost.svg)
"""),

md("""
---
## 4. §3.1 — why the deep configurations needed pre-initialisation

**⏱ needs a GPU and Imagenette.** Reproduce with `python init_study.py --data path/to/imagenette2`.

> "The initialisation of the network weights is important, since bad initialisation can stall learning
> due to the instability of gradient in deep nets. To circumvent this problem, we began with training
> the configuration A ... Then, when training deeper architectures, we initialised the first four
> convolutional layers and the last three fully-connected layers with the layers of net A."

Read once, that is an engineering workaround. Run it and it is the difference between a network that
learns and one that does not.
"""),

code("""
h = json.load(open("results/init_study.json"))["heatmap"]
print(f"Configuration E, {h['epochs']} epochs on Imagenette. Chance is {h['chance_pct']}%.\\n")
print(f"{'initialisation':<22}" + "".join(f"{lr:>9.0e}" for lr in h["learning_rates"]))
for row in h["rows"]:
    cells = "".join("      NaN" if n else f"{a:9.1f}" for a, n in zip(row["acc"], row["nan"]))
    print(f"{row['init']:<22}{cells}")
s = h["summary"]
print(f"\\n{s['cells_that_train']} cells train, {s['cells_near_chance']} sit at chance, "
      f"{s['cells_diverged']} diverge.")
"""),

md("""
Two cells out of twenty. Only Kaiming, and only at the two lowest rates — and its window has two
*different* edges: at 1e-2 the network settles at chance, from 3e-2 upward it blows up. Depth narrows
the window from both sides at once.

Glorot gives half Kaiming's variance per layer, so the shortfall in standard deviation compounds as a
power of the depth.
"""),

code("""
sd = json.load(open("results/init_study.json"))["signal_deficit"]
for b in sd["bars"]:
    print(f"{b['case']:<16}{b['init']:<12}{b['layers']:>3} layers   "
          f"{b['deficit']:>9,}x   measured {b['measured_val_acc']:>5}%")
print(f"\\nthreshold between training and not: between {sd['threshold_band'][0]}x "
      f"and {sd['threshold_band'][1]}x")
ck = sd["checked_against_measurement"]
print(f"\\n{ck['reading']}")
"""),

md("""
Survivable over 8 layers, not over 16. The paper's transplant removes four bad layers, and four is enough.

**One part could not be reproduced.** The paper trains its donor A with its own random initialisation —
and that initialisation does not train configuration A here either, at any rate tried. So the paper's
donor does not exist to copy from. The donor used is Kaiming-trained, which makes this a test of whether
a *trained A* rescues E, not a test of the paper's method end to end.

The deficit figures are **modelled**, and one of them has been checked against a measurement: the model
understates it six-fold. Read that axis as an ordering, not a quantity.

![](figures/init.svg)
"""),

md("""
---
## 5. §3.2 — the one layer that ever needed 224

Every description of VGG begins by saying it takes a 224×224 input. Only one layer in it ever required
that. The convolutions never cared — they slide, and a bigger input just gives a bigger output. The
constraint is the first fully-connected layer, which demands exactly 7·7·512 = 25,088 numbers.

§3.2 removes that one constraint:

> "the fully-connected layers are first converted to convolutional layers (the first FC layer to a
> 7 × 7 conv. layer, the last two FC layers to 1 × 1 conv. layers)"
"""),

code("""
fc = nn.Linear(25088, 4096)
cv = nn.Conv2d(512, 4096, kernel_size=7)
print(f"Linear weight {tuple(fc.weight.shape)}  ->  {fc.weight.numel():,} values")
print(f"Conv2d weight {tuple(cv.weight.shape)}  ->  {cv.weight.numel():,} values")
print(f"\\n512 * 7 * 7 = {512*7*7:,}   the same numbers, folded differently")
"""),

md("""
The reshape is legal only because the flatten feeding the `Linear` layer runs in C order over a
(512, 7, 7) tensor, so column `c*49 + h*7 + w` of the matrix is element `(c, h, w)` of the kernel. Under
any other flatten order `.view()` would still succeed, raise nothing, and place every weight somewhere
wrong — the network would run and return confident nonsense.

So the conversion is asserted, not assumed:
"""),

code("""
from vgg import to_dense, score_map_size

torch.manual_seed(0)
model = make_vgg("D").eval()
dense = to_dense(model)

x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    gap = (model(x) - dense(x).flatten(1)).abs().max().item()
print(f"max disagreement at 224: {gap:.3e}   (float32 round-off, not an approximation)")
print("parameter count preserved:",
      sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in dense.parameters()))
"""),

md("""
Exactly zero would be the suspicious answer: a matrix multiply and a convolution accumulate in different
orders.

Now the payoff. Once the classifier is a convolution it can slide, and a larger input produces a **class
score map** instead of a single prediction.
"""),

code("""
print(f"{'input':>12}{'score map':>14}{'predicted':>12}{'positions':>11}")
for h_, w_ in [(224, 224), (256, 256), (384, 384), (512, 512), (256, 341)]:
    with torch.no_grad():
        got = tuple(dense(torch.randn(1, 3, h_, w_)).shape[2:])
    want = score_map_size(h_, w_)
    assert got == want
    print(f"{h_:>6} x {w_:<4}{got[0]:>6} x {got[1]:<6}{want[0]:>5} x {want[1]:<6}{got[0]*got[1]:>10}")
"""),

md("""
At 224 the map is a single position — dense evaluation there *is* the original network. The last row is
the realistic case: the paper rescales the smaller side to Q and leaves the aspect ratio alone, so real
images give rectangular maps.

**The network's stride is 32**, so each score-map position is a 224×224 window offset 32 pixels from the
last. Dense evaluation is multi-crop at every 32-pixel offset — for the cost of one pass, because the
convolutional work underneath every window is shared instead of recomputed.

![](figures/conversion.svg)
"""),

md("""
---
## 6. Table 5, on the paper's own validation set

**⏱ needs ImageNet.** All 50,000 ILSVRC-2012 validation images, torchvision's pretrained VGG-16,
float32. Reproduce with `python dense_eval.py --data path/to/imagenet/val`.

Before any comparison, the pipeline was checked against something external — torchvision publishes a
top-1 accuracy for these exact weights under exactly this protocol.
"""),

code("""
d = json.load(open("results/dense_eval.json"))
pc = d["pipeline_check"]
print(f"measured      {pc['measured_top1_error_pct']}% top-1 error")
print(f"torchvision   {pc['torchvision_reports_pct']}%")
print(f"gap           {pc['gap_pct']}%  ({pc['gap_images']} images in 50,000)")
"""),

md("""
Every other check in this study is internal — comparing our numbers to our other numbers. This one is not.

Now the three protocols, posteriors averaged over Q = {256, 384, 512} as §4.2 specifies:
"""),

code("""
ms = d["multi_scale"]
print(f"{'protocol':<20}{'top-1':>9}{'top-5':>9}{'paper':>16}{'offset':>9}")
for k in ["dense", "multicrop", "multicrop & dense"]:
    o, p = ms["measured"][k], ms["paper_table_5_config_D"][k]
    print(f"{k:<20}{o['top1']:>8.3f}%{o['top5']:>8.3f}%"
          f"{p['top1']:>9} / {p['top5']:<4}{ms['offset_from_paper_top1'][k]:>+8.3f}")

g = ms["gaps"]["dense_to_combination"]
print(f"\\ndense to combination: {g['measured']} here, {g['paper']} in the paper")
"""),

md("""
The ordering reproduces and so do the gaps. The ~0.45 point offset is the **weights**, not the protocol —
torchvision trained these with `RandomResizedCrop` rather than the paper's scale jittering, so there is
no training scale S at all. The evidence that it is the weights is that the offset barely moves across
three protocols: 0.442, 0.500, 0.412. Protocol and weight quality add rather than interact.
"""),

code("""
fam = d["mcnemar"]["families"][0]
print(f"{fam['name']} — {fam['tests']} tests declared in advance, "
      f"Bonferroni threshold {fam['bonferroni_alpha']}\\n")
for r in fam["results"]:
    verdict = "significant" if r["p"] < fam["bonferroni_alpha"] else ""
    print(f"  {r['metric']}  {r['a']:<18} vs {r['b']:<18} "
          f"{r['only_a']:>5} / {r['only_b']:<5} p = {r['p']:.4f}  {verdict}")
print(f"\\n{fam['verdict']}")
"""),

md("""
![](figures/table5.svg)
"""),

md("""
---
## 7. Why every comparison above is a paired test

This is the part that nearly went wrong.

At a single scale, dense evaluation and centre-cropping score 14.73% and 14.88% top-1 on the same 3,925
images. A difference of 0.15 points — six images. Easy to explain. The explanation would have been fiction.
"""),

code("""
w = d["where_dense_acts"]
m = w["matrix_rows_crop_cols_dense"]
names = w["buckets"]

print("rank of the correct class — rows centre-crop, columns dense\\n")
print(" " * 12 + "".join(f"{n:>11}" for n in names))
for i, row in enumerate(m):
    print(f"{names[i]:<12}" + "".join(f"{v:>11,}" for v in row))

print(f"\\ntop-1  {w['top1']['crop']}% vs {w['top1']['dense']}%   "
      f"McNemar {w['top1']['mcnemar'][0]} / {w['top1']['mcnemar'][1]}, "
      f"p = {w['top1']['mcnemar'][2]}")
print(f"top-5  {w['top5']['crop']}% vs {w['top5']['dense']}%   "
      f"McNemar {w['top5']['mcnemar'][0]} / {w['top5']['mcnemar'][1]}, "
      f"p = {w['top5']['mcnemar'][2]}  significant")
"""),

md("""
The two protocols **disagree on 196 images**. Dense gets 95 right that cropping misses; cropping gets 101
right that dense misses. They cancel, and the aggregate difference hides both facts — that neither is
better, and that they are very different predictors.

Which is precisely why the paper can call them *complementary* and why combining them beats each alone.
The disagreement is the raw material of the combination.

Two more things fall out of that matrix:

- Of the 38 images dense pulls into the top five, **28 land at ranks 2–5**, not rank 1. Dense improves
  the candidate list, not the top slot — and top-1 scores rank 2 and rank 900 identically, so it reports
  nothing.
- Of the 3,347 images the centre crop already ranked first, **none** fell below rank five. Averaging over
  windows dilutes a confident right answer; it does not destroy one. (A plainer reading competes: those
  are the easy images, and easy images survive any window. The two were not separated.)

### And the reason this was run on 50,000 images and not 3,925
"""),

code("""
ss = d["sample_size"]
print(f"{'comparison':<34}{'3,925 images':>22}{'50,000 images':>24}")
for c in ss["comparisons"]:
    s_, l_ = c["small"], c["large"]
    print(f"{c['test'].replace(', top-1',''):<34}"
          f"{s_[0]:>8} / {s_[1]:<4} p={s_[2]:.2f}"
          f"{l_[0]:>11} / {l_[1]:<5} p={l_[2]:.4f}")
pw = ss["power"]
print(f"\\n{pw['discordant_pairs_available']} discordant pairs available, "
      f"about {pw['discordant_pairs_needed']} needed -> ~{pw['images_needed']:,} images")
print(pw["reading"])
"""),

md("""
Identical code, identical weights, identical protocols. Only the sample changes. The smaller set was not
misleading — it was **silent**, and the two look the same if only one of them is ever run.
"""),

md("""
---
## 8. Which lever actually moves the number
"""),

code("""
print(f"{'change':<44}{'points':>9}{'p':>10}")
for lv in d["two_levers"]["levers"]:
    name = lv["change"] + ("" if lv["on"] == "-" else f",  {lv['on']}")
    flag = "" if lv["significant"] else "   not significant"
    print(f"{name:<44}{lv['points']:>9.3f}{lv['p']:>10.4f}{flag}")
"""),

md("""
§4.3 and a table go to choosing multi-crop over dense — worth **0.142 points**, and it does not survive
its own threshold. Averaging the same model over three test scales is worth **1.01 to 1.13 points**.

Both results are in the paper. Table 4 has the scale gain and Table 5 the protocol gain, but they sit in
different tables with different baselines and nothing invites the reader to put them side by side. Doing
so reverses which one looks important.

![](figures/levers.svg)
"""),

md("""
---
## What was and was not reproduced

**Reproduced**

- Table 1 and Table 2 — all six configurations, parameter counts exact
- §3.2's FC-to-convolutional conversion, to float32 round-off
- Score-map geometry, predicted then measured
- Table 5's ordering and gaps, on the paper's own validation set
- Table 4's claim that test-time scale jittering helps
- Table 5's implicit claim that dense and multi-crop are near-equivalent alone

**Not reproduced**

- **Training.** No network here was trained to convergence on ImageNet.
- **The paper's own initialisation** — it does not train configuration A here, so the donor its
  pre-training procedure needs could not be produced.
- **Table 4's Q = {S−32, S, S+32} finding** — these weights have no training scale S.
- **§4.4 ConvNet fusion.**

**Measured here, not in the paper**

- What A-LRN actually costs in time and memory
- Milliseconds per GMAC for each step up Table 1
- That the protocol's benefit is nearly independent of weight quality
- That the scale lever is roughly three times the protocol lever
- That dense evaluation improves the candidate list rather than the top slot

Every number above comes from `results/*.json`, written by the scripts in this folder. The figures are
drawn from the same files by `figures.py`, which measures nothing — so they regenerate on a laptop, and
each one traces back to a line in a file rather than to something typed into a plotting script once.
"""),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    args = ap.parse_args()

    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })

    if not args.no_exec:
        from nbclient import NotebookClient
        print("executing...")
        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": HERE}},
                       allow_errors=False).execute()

    path = os.path.join(HERE, "VGG.ipynb")
    nbf.write(nb, path)
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown")
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"wrote VGG.ipynb — {n_md} markdown, {n_code} code, {n_out} with captured output")


if __name__ == "__main__":
    main()
