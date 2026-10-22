# papers-from-scratch

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![Papers implemented](https://img.shields.io/badge/papers-3-informational?style=flat-square)](#implemented-papers)

This repo holds from-scratch PyTorch reimplementations of foundational computer
vision papers, built to match what each paper describes rather than what later
frameworks made convenient. Faithfulness matters because the convenient version
is often a different model. Common implementations of AlexNet get its
parameter count wrong, apply its normalisation five times weaker than the paper
specifies, and drop the wrong activations. Common implementations of LeNet-5
swap its trainable subsampling for plain average pooling, ignore C3's partial
connection table, and pick the predicted class with `argmax` from what are
actually distances. Common implementations of VGG's dense evaluation leave
torchvision's adaptive pool in place, which collapses the score map to 1×1 and
turns the procedure back into the single-crop evaluation it was meant to
replace — without raising an error. Each implementation here checks its counts
against the paper's own figures, and each README lists the details that are easy
to get wrong and every place the code still departs from the paper.

## Implemented papers

| Model | Paper | Authors | Year | Code |
|---|---|---|---|---|
| LeNet-5 | [Gradient-Based Learning Applied to Document Recognition](https://doi.org/10.1109/5.726791) | LeCun, Bottou, Bengio, Haffner | 1998 | [`lenet5/`](lenet5/) |
| AlexNet | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | Krizhevsky, Sutskever, Hinton | 2012 | [`alexnet/`](alexnet/) |
| VGG | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | Simonyan, Zisserman | 2015 | [`vgg/`](vgg/) |

The LeNet-5 and AlexNet papers are not on arXiv; those links go to the official
publication pages. Paper PDFs are not committed to this repo.

## Verification

All counts are from `torchinfo` output, not computed by hand.

| Model | Quantity | This repo | Paper | Match |
|---|---|---:|---:|---|
| LeNet-5 | Parameters | 60,000 | 60,000 (§II.A) | Exact |
| LeNet-5 | Per-layer parameters | 156 / 12 / 1,516 / 32 / 48,120 / 10,164 | same (§II.B) | Exact, 6 of 6 layers |
| LeNet-5 | Connections | 340,908 | 340,908 (§II.A) | Exact, once the output layer's 840 connections are included |
| AlexNet | Parameters | 60,965,224 | "60 million" (abstract) | Consistent (the paper gives only the rounded total) |
| VGG | Parameters, all six configs | 132,863,336 / 132,863,336 / 133,047,848 / 133,638,952 / 138,357,544 / 143,667,240 | 133 / 133 / 133 / 134 / 138 / 144 million (Table 2) | Exact, 6 of 6 configurations |
| VGG | Weight layers per config | 11 / 11 / 13 / 16 / 16 / 19 | same (Table 1) | Exact, 6 of 6 |
| VGG | FC-to-conv conversion (§3.2) | agrees with the original network to 1.9e-08 | exact rewriting | Float32 round-off only |
| VGG | Pipeline against an external reference | 28.418% top-1 error | 28.408% reported by torchvision for these weights | 5 images in 50,000 |
| VGG | Table 5 ordering, 50,000 ILSVRC images | 25.242 / 25.100 / 24.812 top-1 | 24.8 / 24.6 / 24.4 | Ordering and gaps reproduce; see [`vgg/`](vgg/README.md#the-evaluation-study) |

Each model's README has the layer-by-layer breakdown.

## Scale: seventeen years apart

| | Parameters | Mult-adds (`torchinfo`) | Connections (full count) |
|---|---:|---:|---:|
| LeNet-5 (1998) | 60,000 | 332,188 (0.33 M) | 340,908 |
| AlexNet (2012) | 60,965,224 | 725,066,088 (725.07 M) | 725,066,088 |
| VGG-16, config D (2015) | 138,357,544 | 15,470,264,320 (15.47 G) | 15,470,264,320 |
| **LeNet-5 → AlexNet** | **1,016×** | **2,183×** | **2,127×** |
| **AlexNet → VGG-16** | **2.27×** | **21.34×** | **21.34×** |

The last row is the VGG paper's whole argument as a pair of numbers. Three years
after AlexNet, VGG carries 2.27× the parameters and does 21.34× the arithmetic.
Depth buys compute, not weights — the parameters sit in the fully-connected
layers, which barely changed, while the convolutional stack grew from 2.33 M to
14.7 M parameters and from 8% to 99.2% of the mult-adds.

There are two compute ratios because `torchinfo` undercounts LeNet-5. It
only counts built-in `Conv2d` and `Linear` modules. It cannot see inside
LeNet-5's custom S2 and S4 subsampling layers or the RBF output layer, which
together contribute 7,880 + 840 connections. Its 332,188 is therefore low, and
dividing by it gives the larger ratio, 2,183×. The full count of 340,908 is
the figure the paper states, and gives 2,127×. AlexNet is built entirely from
standard modules, so `torchinfo`'s count for it is complete. Both ratios use
exact totals. Dividing the rounded figures (725.07 / 0.33) gives 2,197×, which
is wrong.

## Setup

```bash
git clone https://github.com/Mohd-Abdul-Rafay/papers-from-scratch.git
cd papers-from-scratch
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python lenet5/lenet5.py
python alexnet/alexnet.py
python vgg/vgg.py
python make_diagrams.py     # regenerates lenet5/lenet5.{png,svg}, alexnet/alexnet.{png,svg}
```

`vgg/dense_eval.py` is the one script that needs data — the ILSVRC-2012
validation set as 1,000 wnid-named folders:

```bash
python vgg/dense_eval.py --data /path/to/imagenet/val
python vgg/dense_eval.py --data ... --scales 256 --protocols crop dense   # quicker
```

`make_diagrams.py` needs Inter Tight and JetBrains Mono installed (both SIL OFL)
and prints a loud warning if either is missing -- without them the diagrams
regenerate in fallback fonts and every SVG shows up as a diff.

Each script prints a layer-by-layer parameter report against the paper and runs
a forward-pass shape check. Tested with Python 3.14, torch 2.14, torchvision 0.29,
torchinfo 1.8 and matplotlib 3.11. Datasets go under `data/`, which is git-ignored.

## What this repo checks

Five findings that come from checking the code against the papers rather
than against other implementations:

1. **LeNet-5's 340,908 connections reconcile exactly.** The paper's
   per-layer connection figures (§II.B) sum to 340,068, which is 840 short
   of the total it states in §II.A. The gap is exactly the output RBF layer
   (84 inputs × 10 classes), which §II.B never counts. With it included,
   every figure in the paper agrees. [Derivation](lenet5/README.md#connections-reconciling-the-papers-total)
2. **AlexNet's LRN sits before max-pooling.** §3.5 puts response
   normalisation directly after the ReLU of conv1 and conv2, with pooling
   after it. The script checks the order at runtime. Separately, PyTorch's
   `LocalResponseNorm` divides `alpha` by the window size, so the paper's
   α = 1e-4 must be passed as `alpha=5e-4`. Numerically verified. [Details](alexnet/README.md#details-most-implementations-get-wrong)
3. **AlexNet's dropout drops the outputs of fc1 and fc2.** §4.2 applies it
   to the outputs of the first two fully-connected layers, which hold 89.4%
   of the parameters. The common torchvision ordering (Dropout → Linear)
   instead drops the 9,216 conv features, which the paper never drops, and
   leaves fc2's output undropped. [Details](alexnet/README.md#details-most-implementations-get-wrong)
4. **VGG's 3×3 argument is an argument about the ReLUs.** Two stacked 3×3
   convolutions with no activation between them collapse exactly into one 5×5
   convolution — convolution is linear. The depth claim in §2.3 only survives
   because of the non-linearity between the layers, which the cost comparison
   (27C² against 49C²) does not mention. [Details](vgg/README.md#why-33-stacks)
5. **Dense evaluation's gain is in the candidate list, not the top slot.**
   Measured over all 50,000 ILSVRC validation images, dense evaluation beats
   centre-cropping clearly on top-5 and not at all on top-1 at a single scale,
   and it never pushes a correct answer out of the top five. Separately,
   averaging over three test scales is worth about three times what choosing
   multi-crop over dense is worth — the reverse of the attention the paper gives
   them. [Details](vgg/README.md#three-results-not-in-the-paper)

## License

MIT. See [LICENSE](LICENSE).
