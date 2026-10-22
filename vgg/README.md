# VGG

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) —
Karen Simonyan, Andrew Zisserman, ICLR 2015.

Two files:

| File | What it does |
|---|---|
| [`vgg.py`](vgg.py) | All six Table 1 configurations, checked against Table 2; the Section 3.2 FC-to-convolutional conversion, checked against the original network |
| [`dense_eval.py`](dense_eval.py) | Sections 3.2 / 4.2 / 4.3 on the ILSVRC-2012 validation set: dense, multi-crop, their combination, across test scales |

```bash
python vgg.py                                      # architecture checks, no data needed
python dense_eval.py --data /path/to/imagenet/val  # the evaluation study
```

## Architecture

Depth is the only variable. All six configurations share one receptive field
(3×3), one stride (1), one padding rule, the same five max-pool stages and the
same three fully-connected layers. That is the whole design: hold everything
constant, vary depth, and the error difference can be attributed to depth.

| Config | Weight layers | Parameters |
|---|---:|---:|
| A | 11 | 132,863,336 |
| A-LRN | 11 | 132,863,336 |
| B | 13 | 133,047,848 |
| C | 16 | 133,638,952 |
| D | 16 | 138,357,544 |
| E | 19 | 143,667,240 |

A and A-LRN are the same size because local response normalisation has no
parameters. A-LRN is a control, not a model — §4.1 finds it does not improve on
A, and it is dropped from B onward.

### Why 3×3 stacks

Three stacked 3×3 layers see the same 7×7 receptive field as one 7×7 layer, but
over C channels they cost 27C² weights against 49C² — the single layer is 81%
more expensive. The stack also applies three non-linearities instead of one.

That second point is doing all the work, and it is easy to read past.
Convolution is linear, so **two stacked 3×3 convolutions with no activation
between them collapse exactly into one 5×5 convolution** — same function, same
output, provably. The depth argument is an argument about the ReLUs, not about
the kernels. Remove them and the stack is not deep, it is just a more awkward
way of writing a wider kernel.

### The parameter/compute inversion

For configuration D:

```
CONV     14,714,688 params (10.6%)   15,346,630,656 mult-adds (99.2%)
FC      123,642,856 params (89.4%)      123,633,664 mult-adds ( 0.8%)
```

A convolutional weight is reused at every spatial position; a fully-connected
weight is used once. So the conv stack does essentially all the arithmetic while
holding a tenth of the weights, and the FC layers hold nine tenths of the
weights while doing under one percent of the work. The FC layers are where the
memory goes; the conv stack is where the time goes.

The FC parameter and mult-add counts differ by exactly 9,192 — the biases.

### Dense evaluation (§3.2)

Only one thing in VGG ever required a 224×224 input: the first fully-connected
layer, which demands exactly 7·7·512 = 25,088 numbers. The convolutions never
cared about input size at all.

So §3.2 removes that one constraint. A `Linear(25088, 4096)` holds its 102.7
million weights in the same order as a `Conv2d(512, 4096, kernel_size=7)` — the
conversion reshapes, it does not retrain. Once it is a convolution it can slide,
and a larger input produces a **class score map** instead of one prediction. The
map is averaged to a fixed-size score vector.

Because the network's stride is 32, each score-map position corresponds to a
224×224 window of the input offset 32 pixels from the last. Dense evaluation is
therefore multi-crop at every 32-pixel offset — for the cost of a single pass,
since the convolutional work underneath is shared.

```
       input     score map    positions
  224 x 224       1 x 1               1     identical to the original network
  256 x 256       2 x 2               4
  384 x 384       6 x 6              36
  512 x 512      10 x 10            100
  256 x 341       2 x 4               8     real images are not square
```

## Verification

`python vgg.py` output, unedited:

```
Table 1 configurations against Table 2

A  —  11 weight layers, Table 1 says 11   5 pools
  conv     9,220,480  ( 6.9%)
  fc     123,642,856  (93.1%)
  total  132,863,336   Table 2: 132,863,336   exact

A-LRN  —  11 weight layers, Table 1 says 11   5 pools
  conv     9,220,480  ( 6.9%)
  fc     123,642,856  (93.1%)
  total  132,863,336   Table 2: 132,863,336   exact

B  —  13 weight layers, Table 1 says 13   5 pools
  conv     9,404,992  ( 7.1%)
  fc     123,642,856  (92.9%)
  total  133,047,848   Table 2: 133,047,848   exact

C  —  16 weight layers, Table 1 says 16   5 pools
  conv     9,996,096  ( 7.5%)
  fc     123,642,856  (92.5%)
  total  133,638,952   Table 2: 133,638,952   exact

D  —  16 weight layers, Table 1 says 16   5 pools
  conv    14,714,688  (10.6%)
  fc     123,642,856  (89.4%)
  total  138,357,544   Table 2: 138,357,544   exact

E  —  19 weight layers, Table 1 says 19   5 pools
  conv    20,024,384  (13.9%)
  fc     123,642,856  (86.1%)
  total  143,667,240   Table 2: 143,667,240   exact

A and A-LRN identical in size: True   LRN carries no parameters
FC-to-conv agrees at 224:     1.863e-08   float32 round-off, not an approximation
conversion preserves count:   True

       input     score map     predicted   positions
  224 x 224       1 x 1         1 x 1              1
  256 x 256       2 x 2         2 x 2              4
  384 x 384       6 x 6         6 x 6             36
  512 x 512      10 x 10       10 x 10           100
  256 x 341       2 x 4         2 x 4              8

all checks passed
```

### The evaluation study

`dense_eval.py` on all 50,000 ILSVRC-2012 validation images, using torchvision's
pretrained VGG-16 (configuration D), Q = {256, 384, 512}:

| Protocol | top-1 | top-5 | Paper, Table 5 |
|---|---:|---:|---|
| dense | 25.242% | 7.764% | 24.8 / 7.5 |
| multi-crop | 25.100% | 7.676% | 24.6 / 7.5 |
| multi-crop & dense | **24.812%** | **7.420%** | 24.4 / 7.2 |

The ordering reproduces. So do the gaps: dense to the combination is 0.430 here
against the paper's 0.4.

Pre-registered McNemar tests, six comparisons, Bonferroni threshold 0.0083:

| Comparison | p | Verdict |
|---|---:|---|
| top-1 dense vs multi-crop | 0.1448 | not distinguishable |
| top-5 dense vs multi-crop | 0.2464 | not distinguishable |
| top-1 dense vs combination | 0.0000 | **combination better** |
| top-5 dense vs combination | 0.0000 | **combination better** |
| top-1 multi-crop vs combination | 0.0000 | **combination better** |
| top-5 multi-crop vs combination | 0.0000 | **combination better** |

The paper's headline claim — that the two are complementary and their
combination beats each alone — holds on both metrics. That dense and multi-crop
cannot be told apart is also consistent with the paper, whose own top-5 figures
for the two are identical at 7.5.

The pipeline was checked against an external reference before any of this was
run: centre-crop without flipping gives 28.418% top-1 error against the 28.408%
torchvision reports for these weights, a gap of five images in fifty thousand.

### Three results not in the paper

**The protocol's benefit is nearly independent of the weights.** These weights
are about half a point behind the paper's, and the offset barely moves across
protocols — 0.442, 0.500, 0.412. Evaluation protocol and weight quality add
rather than interact.

**The scale lever is about three times the protocol lever.** Averaging over
Q = {256, 384, 512} instead of Q = 256 alone is worth 1.01–1.13 points of top-1
(p = 0.0000 for both dense and multi-crop). Choosing multi-crop over dense is
worth 0.14 points and is not significant. §4.3 and a table go to the smaller of
the two.

**Dense evaluation improves the candidate list more than the top slot.** At a
single scale, dense beats centre-crop clearly on top-5 (p = 0.0022) and not at
all on top-1 (p = 0.72) — and of the images it pulls into the top five, 28 of 38
land at ranks 2–5 rather than rank 1. Meanwhile it never pushed a correct answer
out of the top five at all: of 3,347 images the centre crop ranked first, zero
fell below rank 5 under dense. Averaging over windows dilutes a confident right
answer; it does not destroy one.

## Details most implementations get wrong

**Leaving torchvision's `AdaptiveAvgPool2d((7, 7))` in the dense model.**
torchvision inserts it so its `Linear` classifier accepts any input size. It
solves the same problem §3.2 solves, but by throwing the spatial resolution away
rather than keeping it. Left in, it forces a 1×1 score map at every Q, raises no
error, and silently turns dense evaluation back into single-crop evaluation.

**Averaging in the wrong space.** The score map is averaged *before* the softmax,
in logit space. The original and flipped images are averaged *after* it, in
posterior space. The paper states both; they are different operations in
different places and swapping them changes the answer quietly.

**Assuming `.view()` on the FC weight is safe.** `Linear(25088, 4096).weight.view(4096, 512, 7, 7)`
is correct only because the flatten feeding that layer runs in C order over a
(512, 7, 7) tensor. Under any other ordering the reshape still succeeds and
scrambles the weights. `vgg.py` asserts the converted network reproduces the
original instead of assuming it.

**Normalising with the paper's statistics while using torchvision's weights.**
The paper subtracts the mean only. torchvision's weights were trained with
mean *and* standard deviation. Mixing them costs accuracy and raises nothing.

**Padding.** Every convolution uses `padding = k // 2`, so 3×3 gets 1 and the
1×1 layers in configuration C get 0 from the same rule. Only the max-pools
change spatial size, which is why 224 → 7 is exactly 2⁵.

**Reading A-LRN's `alpha` off the paper directly.** PyTorch's
`LocalResponseNorm` divides alpha by `size` internally and the paper's formula
does not, so the paper's 1e-4 is passed as 5e-4.

**Treating a 0.2-point difference as a result.** At 3,925 validation images
none of the protocol comparisons here reach significance; at 50,000 almost all
of them do. Worse, a small aggregate difference can hide a large number of real
disagreements that cancel — dense and centre-crop differ on about 5% of images
while scoring almost identically. `dense_eval.py` reports paired McNemar tests
for this reason, and the comparisons are fixed before the numbers are seen.

## Caveats

* `dense_eval.py` evaluates **torchvision's** pretrained VGG-16, not weights
  trained here. torchvision trained it with `RandomResizedCrop(224)` and
  `--lr 1e-2` for 90 epochs, not with the paper's scale jittering over
  S ∈ [256, 512], so there is no well-defined training scale S. That is why the
  absolute errors sit about 0.45 points above the paper's, and why Table 4's
  Q = {S−32, S, S+32} finding cannot be reproduced here — there is no S.

* Training is not reproduced. The paper's own initialisation procedure,
  pre-initialising the deeper nets from a trained configuration A, is not
  implemented in these files.

* §4.4 ConvNet fusion — ensembling several trained networks — is not
  implemented.

* `vgg.py` builds and checks the architectures; it does not train them. The
  parameter counts assume 1,000 classes and the paper's 224×224 input.

* The multi-crop grid uses `torch.linspace` over the available offsets, which
  matches "5 × 5 regular grid" but is not pinned to a specific crop placement by
  the paper. Different reasonable placements give slightly different numbers.
