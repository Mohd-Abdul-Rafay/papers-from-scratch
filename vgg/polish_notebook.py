"""
Add markdown to the working notebook without touching a single cell or output.

VGG.ipynb is the record of the session the study was actually done in: 44 code
cells in the order they were run, carrying the real output of real runs on an
A100 — including the runs that failed, the predictions that turned out wrong,
and the bug an invariant caught. That is the part worth keeping, and none of it
is regenerable: re-executing the notebook here would replace measurements taken
on a GPU with whatever this machine produces.

So nothing is executed and no source is edited. This script reads the original,
inserts markdown cells around the existing ones, and writes the result. Every
code cell and every output in the output file is byte-identical to the input,
and the script asserts that before writing.

    python polish_notebook.py --in <session export>.ipynb --out VGG.ipynb

The input is the raw Colab export. It is not committed here because every one of
its cells is already in VGG.ipynb, byte for byte — that is the guarantee this
script asserts before it writes. Diff the two and the only difference is
markdown.
"""

import argparse
import copy
import json

# Markdown to place BEFORE and AFTER each original code cell, by its index.
# Only cells that need something appear here.
BEFORE = {
 0: """
# VGG, worked through

[*Very Deep Convolutional Networks for Large-Scale Image Recognition*](https://arxiv.org/abs/1409.1556) — Simonyan & Zisserman, ICLR 2015.

This is the notebook the study was done in, in the order it was done. The code and every output below
are exactly as they ran — on a Colab A100 for the GPU work, on Imagenette and then on all 50,000
ILSVRC-2012 validation images. Nothing has been re-executed and nothing has been tidied away, including
the runs that produced NaN, the prediction that was wrong by a factor of thirty, and the configuration
bug that only an invariant caught.

The commentary between the cells was added afterwards. The cells were not.

Clean, runnable versions of this work live beside it: [`vgg.py`](vgg.py) for the architectures,
[`cost.py`](cost.py), [`init_study.py`](init_study.py), [`dense_eval.py`](dense_eval.py) and
[`analysis.py`](analysis.py) for the measurements.

---

## 1. Shapes, before anything else

Everything downstream depends on knowing exactly what size comes out of a convolution, so that is
checked against the formula first rather than assumed.
""",
 6: """
---

## 2. Where the memory goes

VGG's reputation is that it is memory-hungry. Worth finding out where the memory actually sits before
trying to save any.

The obvious lever is `inplace=True` on the ReLUs — it should avoid storing a second copy of every
activation. Predicted saving at batch 32: several hundred megabytes.
""",
 7: """
The saving is zero. That is a real measurement and the obvious conclusions from it are both wrong: the
rewrite did take effect, and the memory was not saved.

The way to find out which is to stop looking at one number and watch memory through the whole step.
""",
 8: """
---

## 3. Where the parameters go

From memory to parameters. Configuration D is the famous VGG-16, and the paper's Table 2 gives it as
"138 million" — rounded, so an exact count either lands inside the rounding or it does not.
""",
 10: """
---

## 4. What local response normalization costs

Table 2 gives A and A-LRN the same parameter count, which is correct: LRN has no parameters. The paper
says only that it "leads to increased memory consumption and computation time", with no figure.

This is a first probe on a cut-down stack, not the full configuration — the full one comes in §5.
""",
 11: """
---

## 5. All six configurations, with the checks written first

Table 1 in code. The important part is not the builder, it is the block of invariants above it: every
fact that must hold if the table is read correctly, asserted before anything is benchmarked.

This is what caught the bug. An earlier version of the builder sliced the configuration list from the
wrong index, which silently skipped one layer and added an extra pool. The network still ran, still
produced plausible numbers, and would have gone into the write-up unnoticed — it crashed nothing. The
invariant on the pool count is what stopped it.
""",
 12: """
---

## 6. Does it learn at all?

Before any accuracy number means anything, the question is whether a given initialisation lets the
network learn *anything*. The reference point is the loss of a network that has learned nothing:
uniform over 1,000 classes is exactly ln(1000) = 6.9078.
""",
 13: """
The gradient profile says which initialisations are broken. It does not say whether the survivors can
train. For that: can the network overfit a single batch? If it cannot memorise sixteen images, nothing
about a real dataset will help it.
""",
 14: """
One learning rate is not a verdict on an initialisation — it is a verdict on that pair. The grid
separates the two.
""",
 15: """
---

## 7. Initialisation on real data

The single-batch test is a toy. Imagenette is 10 classes and ~9,500 training images — still not ILSVRC,
but a real dataset with real validation accuracy, and five epochs of it.

Both A (11 layers) and E (19 layers) are run, because the paper's claim is specifically about what
happens as depth grows.
""",
 16: """
Two learning rates leave gaps. Configuration E is the one that fails, so the gaps get filled on E.
""",
 17: """
---

## 8. The paper's transplant

> "we began with training the configuration A ... Then, when training deeper architectures, we
> initialised the first four convolutional layers and the last three fully-connected layers with the
> layers of net A."

The specification is **by layer**, not by position. A's first four convolutions are conv1_1, conv2_1,
conv3_1, conv3_2 — and in E those sit at different positions, because E has two convolutions in the
early blocks where A has one. Copying A's first four into E's first four positions copies the right
weights into the wrong shapes, so the layers are matched by name.
""",
 19: """
One more reading of §3.1 to settle. The paper specifies "a zero-mean Gaussian distribution with the
1e-2 variance" — and variance 1e-2 is standard deviation 0.1, not 0.01. Read literally:
""",
 20: """
---

## 9. Section 3.2 — the fully-connected layers become convolutions

From training to testing. Everything above was about getting a network to learn; everything below is
about how to ask a trained one for an answer.

The weights here are torchvision's pretrained VGG-16, not ours — the question from here on is about the
evaluation procedure, and that can be asked of any competent configuration D.
""",
 21: """
Only one layer in VGG ever required a 224×224 input: the first fully-connected one, which demands
exactly 7·7·512 = 25,088 numbers. The convolutions never cared.

A `Linear(25088, 4096)` and a `Conv2d(512, 4096, kernel_size=7)` hold the same 102,760,448 weights in
the same order, so the conversion is a reshape. That claim is asserted rather than assumed, because
`.view()` would succeed and scramble the weights under any other flatten order.
""",
 22: """
Once the classifier is a convolution it can slide, and a bigger input gives a map instead of one
answer. Predicted from the formula first, then measured.
""",
 23: """
---

## 10. Dense evaluation, first attempt

Imagenette again, this time for evaluation rather than training. Ten classes, 3,925 validation images.

These are real ImageNet classes and torchvision's weights know all 1,000 of them, so the logits are
kept at full 1,000-way rather than restricted to ten — restricting makes the task so easy that
differences between protocols vanish into noise.
""",
 24: """
The class mapping is the one thing that would be silent if it were wrong, so it is checked against
torchvision's own category names rather than taken on trust. A mapping off by one would not crash; it
would show up as "that class is hard".
""",
 26: """
A sanity check on the weights themselves before comparing anything: torchvision publishes an accuracy
for these, measured under exactly this protocol.
""",
 27: """
Now the actual dense evaluation: whole uncropped image, score map averaged in logit space, soft-max
after, and the flipped image's posteriors averaged in.
""",
 28: """
---

## 11. The comparison was confounded

14.73% against 13.89% looks like dense evaluation winning. It is not a clean comparison: the baseline
had no flip averaging and the dense run did, so the gap mixes two changes.

Four cells of a 2×2 separate them.
""",
 29: """
Aggregate error rates cannot say whether two protocols disagree on two hundred images or on none. A
paired test can: it throws away every image the two agree on, and asks whether the rest split like a
fair coin.
""",
 30: """
That was top-1 only. The top-5 arrays were saved and not used — and the larger differences turn out to
be there.
""",
 31: """
---

## 12. Where dense evaluation actually acts

Top-1 says nothing, top-5 says dense is better. Both can be true if dense moves answers up the list
without moving them to the front. The rank of the correct class under each protocol settles it.
""",
 32: """
Mean log-rank favours dense. Whether that survives a signed test is a different question.
""",
 33: """
---

## 13. Multi-crop, and the paper's actual claim

§4.3: 50 crops per scale, a 5×5 grid with both flips. Table 5's claim is that multi-crop and dense are
complementary — that combining them beats each alone.
""",
 34: """
Which comparisons get tested is decided **before** the numbers are looked at. Running every pair and
reporting the ones under 0.05 is a different and much weaker thing.
""",
 35: """
---

## 14. Getting the paper's own data

Nothing above reached significance. The power calculation said why: about 64 discordant pairs where
roughly 130 were needed — the experiment was about half the size it needed to be for this one claim.

Imagenette is also the wrong regime. Its ten classes are easily separated; the paper sits at 24.8%
top-1 error, this sits at 13.8%. So: the real ILSVRC-2012 validation set, all 50,000 images.
""",
 37: """
The label convention is the thing that would ruin 50,000 images silently, so it is checked before the
7 GB download rather than after. ImageNet class index = position of the wnid in sorted order — checked
for monotonicity on one shard first, then across all 1,000 classes.
""",
 39: """
---

## 15. Precision, then the real run

Half precision would make this faster. Whether it is safe depends on whether its noise is smaller than
the effects being measured — which is a thing to measure, not assume.
""",
 40: """
Before the protocols, the pipeline gets checked against something outside this notebook: torchvision
reports 28.408% top-1 for these weights under centre-crop without flip.
""",
 41: """
All four protocols at a single scale, with the Table 5 comparisons fixed in advance.
""",
 42: """
---

## 16. The paper's own protocol

§4.2 averages posteriors over three test scales. Everything above was Q = 256 alone, which is not what
Table 5 reports — so the last run is the full thing: Q = {256, 384, 512}.
""",
}

AFTER = {
 5: """
The formula and PyTorch agree, `padding = k // 2` holds the size for any odd kernel, and the shape
trace confirms that in VGG only the max-pools change resolution — which is why 224 → 7 is exactly 2⁵.
""",
 6: """
**0.0 MB.** Identical to the last decimal, which rules out measurement noise.
""",
 7: """
There it is. The forward peak with `inplace=False` is **802.5 MB**; the first ReLU is where it doubles
(410.5 → 802.5). But the peak over the whole step is **2080.7 MB**, and it is set *after backward*.

So the saving is real and it is in the forward pass — it is invisible in the whole-step peak because
that peak is set while the stored activations are being consumed anyway. The earlier cell measured the
one number training actually fails on, and for that number the answer genuinely is zero.

(The mechanism, separately: `ReluBackward0` saves `_saved_result`, not `_saved_self` — the backward
pass reads the ReLU's output, so overwriting its input is legal. That explains why the rewrite works.
It does not explain the zero. The staircase does.)
""",
 9: """
138,357,544 — inside the paper's rounding. And the split is the thing to remember: the convolutional
stack holds about a tenth of the weights and does over 99% of the arithmetic, while the
fully-connected layers hold nine tenths and do under one per cent.

A convolutional weight is reused at every spatial position; a fully-connected weight is used once.
""",
 10: """
**+68.3% time and +1995.5 MB**, for a layer with no parameters and no arithmetic.

Note the memory figure. In §5 the same comparison on the *full* configurations gives +1995.3 MB — the
same delta from a much larger network, because what LRN costs depends on the tensor it normalises (the
first convolution's output, 392 MB at this batch) and not on anything after it.
""",
 11: """
All six configurations match Table 2 exactly, and the benchmark is the full-network version of §4:
**27.45 → 46.02 ms (+67.7%)** and **4624.1 → 6619.4 MB (+1995.3 MB)**.

The `ms/GMAC` block is a second finding, not in the paper. Adding 1×1 convolutions costs **4.77 ms per
GMAC**; adding 3×3 convolutions costs **1.66**. A 3×3 kernel does nine multiply-accumulates for every
weight it loads and a 1×1 does one. And the same arithmetic added early (224², 112²) costs 4.12 against
1.66 added late. GMAC count alone does not predict time.
""",
 12: """
Four initialisations, and they fail in four different ways.

`paper_literal` (std 0.1) does not fail quietly — loss **2.47e+11**, activations growing to 1.3e+08.
`paper_common` (std 0.01) sits exactly at 6.9078: the network is not broken, it is *silent*. The
per-layer table shows why — activation RMS falls from 5.1e-02 to 2.4e-07 over twelve layers, and the
gradient with it. There is nothing left at the far end to learn from.

`kaiming` starts at 23.35, which looks worse and is not: it is the only one with signal to work with.
""",
 13: """
Only Kaiming moves. The other three sit at ln(1000) for 150 steps on sixteen images.
""",
 14: """
And here is where one reading would have been wrong. `paper_common` looked dead in the profile — but at
lr=1e-2 it reaches **2.5746**, which is learning. The gradient was tiny, not absent, and a large enough
step makes use of it.

A one-shot gradient measurement predicts how a network starts, not whether it can train. The grid is
the measurement; the profile was a hypothesis.
""",
 15: """
Depth is the whole story in this table. On **A**, Glorot reaches 48.9% and Kaiming 61.8% — both train.
On **E**, the same Glorot collapses to 9.1%, which is chance, while Kaiming still reaches 50.1%.

Nothing changed but eight more layers.
""",
 16: """
Filling the gaps gives configuration E's actual window: Kaiming trains at 1e-3 and 3e-3, reaches chance
at 1e-2, and diverges to NaN from 3e-2 up.

So the window has **two different edges**, failing in two different ways, and it is narrower than one
order of magnitude. Glorot and the paper's init have no window at all here — "loss moved 0.0003" over
five epochs.
""",
 18: """
The row that matters is **glorot, lr=1e-3: 9.1% → 49.9%**, a jump of 40.8 points. Same architecture,
same data, same optimiser; seven layers initialised from a trained A instead of from Glorot.

That is the paper's claim, reproduced.

Two things this does not show, and both need saying. The donor here is **Kaiming**-trained, not trained
with the paper's own initialisation — because that initialisation does not train A either, so the
paper's donor could not be produced. And the donor itself scored 58.3%; retrained, the same
configuration gave 61.8% and 63.6% elsewhere in this session, a spread of 5.3 points. Any transplant
result inside that spread is not a finding. 40.8 points is outside it.
""",
 19: """
NaN at every rate. Variance 1e-2 means std 0.1, and at that scale configuration E does not produce a
number at all — so the paper's wording, read literally, describes an initialisation that cannot be used.
Read as std 0.01 it at least runs, and that is almost certainly what was meant.
""",
 21: """
**1.4e-06** — one or two units in the last place of float32. Exactly zero would be the suspicious
answer, since a matrix multiply and a convolution accumulate in different orders.
""",
 22: """
Every size predicted correctly. At 224 the map is a single position, so dense evaluation there *is* the
original network; at 384 it is 6×6 = 36 positions.

The network's stride is 32, so each position is a 224×224 window offset 32 pixels from the last — dense
evaluation is multi-crop at every 32-pixel offset, for the price of one pass, because the convolutional
work underneath is shared.
""",
 24: """
All ten map to the name torchvision gives that index. 3,925 images.
""",
 25: """
14.73% top-1 at full 1,000-way. Cassette player at 44.0% is not a mapping error — a wrong mapping would
read near 0% — it is ImageNet having `tape player`, `CD player` and `radio` as separate classes.
""",
 26: """
28.408% top-1 reported, and the recipe is `RandomResizedCrop`, not the paper's scale jittering. So
these weights have no training scale S, which is worth remembering when Table 4 comes up.
""",
 27: """
13.89% against the crop baseline's 14.73%. 33 distinct score-map shapes, because real images are not
square.
""",
 28: """
Now the confound is visible. Flip averaging alone is worth 0.46 points under crop. Dense **without**
flip is 14.88% — slightly *worse* than the 14.73% baseline.

So the apparent win was mostly the flip.
""",
 29: """
`p = 0.72`. The 0.15-point difference is noise.

But look at the agreement column: the two protocols disagree on **196 images** — dense gets 95 right
that crop misses, crop gets 101 right that dense misses. Two genuinely different predictors whose wins
cancel. That is exactly why the paper can call them *complementary*.
""",
 30: """
On top-5 dense wins clearly: **15 against 38, p = 0.0022**. Same images, same protocols, opposite
answer from top-1 — because top-1 scores rank 2 and rank 900 identically.
""",
 31: """
The empty cell is the finding. Of the 3,347 images the centre crop ranked first, **zero** fell below
rank five under dense. And of the 38 that entered the top five, 28 landed at ranks 2–5 rather than
rank 1.

Dense improves the candidate list and not the top slot. Averaging over windows dilutes a confident
right answer; it does not destroy one.

(A plainer explanation competes: images the centre crop ranks first are the easy ones, and easy images
survive any window. These runs do not separate the two.)
""",
 32: """
`p = 0.073`. Mean log-rank favoured dense, but 236 improved against 205 worsened — nearly even. The
mean was pulled by magnitude, and the claim does not survive.
""",
 34: """
Both pre-specified tests fail. The combination is ahead in both (34 vs 30, 36 vs 27) but by less than
a coin toss would explain.

This is not evidence against the paper. It is 3,925 images not being enough to decide it.
""",
 37: """
Monotonic, and the spot checks line up — `n01491361 → 3 → tiger shark`. The convention holds.
""",
 38: """
50,000 images, 1,000 classes, and the stronger check passes: label *i* is the *i*-th wnid in sorted
order, across all 1,000.
""",
 39: """
**53 images** change their top-1 under fp16 — 0.106%, the same size as the effects being measured. And
fp32 took 59s against fp16's 57s, which is noise: the bottleneck is JPEG decoding, not the GPU.

So half precision was buying nothing and would have cost the result. Everything below is fp32.
""",
 40: """
**28.418% against torchvision's 28.408%** — five images in fifty thousand. The whole pipeline (data,
labels, normalisation, model) now agrees with something measured outside this notebook.
""",
 41: """
At a single scale the ordering already matches Table 5, and now the tests have the power to say so:
crop → dense → multi-crop → combination, with every comparison against crop at p = 0.0000.
""",
 42: """
**24.812% / 7.420%** for the combination, against the paper's 24.4 / 7.2.

The ordering reproduces and so do the gaps — dense to the combination is 0.430 here against the
paper's 0.4. The ~0.45 point offset is the weights, not the protocol: it barely moves across the three
(0.442, 0.500, 0.412), which makes protocol quality and weight quality additive rather than
interacting.

Both pre-specified families pass. And the two levers can finally be put side by side: averaging over
three test scales is worth **1.01–1.13 points**, while choosing multi-crop over dense is worth
**0.142** and does not reach significance. The paper gives §4.3 and a table to the smaller of the two.

---

## What this reproduced, and what it did not

**Reproduced** — Table 1 and Table 2 for all six configurations; §3.2's conversion to float32
round-off; §3.1's pre-initialisation claim; Table 5's ordering and gaps on the paper's own validation
set; Table 4's claim that test-time scale jittering helps.

**Not reproduced** — no network here was trained to convergence on ImageNet; the paper's own
initialisation does not train configuration A, so the donor its procedure requires could not be
produced; Table 4's Q = {S−32, S, S+32} finding needs a training scale S that these weights do not
have; §4.4 ConvNet fusion was not attempted.

**Measured here and not in the paper** — what A-LRN costs in time and memory; milliseconds per GMAC
for each step up Table 1; that the evaluation protocol's benefit is nearly independent of weight
quality; that the scale lever is about three times the protocol lever; and that dense evaluation
improves the candidate list rather than the top slot.
""",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    nb = json.load(open(args.src, encoding="utf8"))
    original = [c for c in nb["cells"]]
    code_in = [c for c in original if c["cell_type"] == "code"]

    cells = []
    dropped = 0
    for i, cell in enumerate(original):
        src = "".join(cell["source"]).strip()
        if cell["cell_type"] == "code" and not src:
            dropped += 1                      # the trailing empty cell
            continue
        for text in (BEFORE.get(i),):
            if text:
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": text.strip("\n").splitlines(keepends=True)})
        cells.append(copy.deepcopy(cell))
        for text in (AFTER.get(i),):
            if text:
                cells.append({"cell_type": "markdown", "metadata": {},
                              "source": text.strip("\n").splitlines(keepends=True)})

    out = dict(nb)
    out["cells"] = cells

    # the guarantee: every code cell and every output is untouched
    code_out = [c for c in cells if c["cell_type"] == "code"]
    assert len(code_out) == len(code_in) - dropped, "a code cell went missing"
    for a, b in zip(code_in[:len(code_out)] if dropped == 0 else
                    [c for c in code_in if "".join(c["source"]).strip()], code_out):
        assert a["source"] == b["source"], "a code cell's source changed"
        assert a.get("outputs") == b.get("outputs"), "a cell's output changed"
    print(f"code cells: {len(code_out)} (dropped {dropped} empty)")
    print(f"markdown cells added: {sum(1 for c in cells if c['cell_type']=='markdown')}")
    print("every code cell and every output byte-identical to the original: True")

    json.dump(out, open(args.dst, "w", encoding="utf8"), indent=1)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
