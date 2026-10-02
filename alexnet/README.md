# AlexNet

**Paper:** A. Krizhevsky, I. Sutskever, G. E. Hinton.
*ImageNet Classification with Deep Convolutional Neural Networks.*
Advances in Neural Information Processing Systems 25 (NIPS 2012).
[NeurIPS proceedings page](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
(not on arXiv)

```bash
python alexnet/alexnet.py   # parameter report, shape trace, LRN-order check
```

## Architecture

Input is 3 × 224 × 224. Output shapes, params and connections are from
`torchinfo`. "Connections" means mult-adds per image, counting each bias as
one connection, as LeNet-5's paper does.

| Layer | Operation | Output shape | Params | Connections |
|---|---|---|---:|---:|
| conv1 | 11×11 / 4, pad 2, 96 maps → ReLU → LRN → maxpool 3/2 | 96 × 27 × 27 | 34,944 | 105,705,600 |
| conv2 | 5×5, pad 2, 256 maps, **groups=2** → ReLU → LRN → maxpool 3/2 | 256 × 13 × 13 | 307,456 | 224,135,424 |
| conv3 | 3×3, pad 1, 384 maps (ungrouped) → ReLU | 384 × 13 × 13 | 885,120 | 149,585,280 |
| conv4 | 3×3, pad 1, 384 maps, **groups=2** → ReLU | 384 × 13 × 13 | 663,936 | 112,205,184 |
| conv5 | 3×3, pad 1, 256 maps, **groups=2** → ReLU → maxpool 3/2 | 256 × 6 × 6 | 442,624 | 74,803,456 |
| fc1 | 9216 → 4096 → ReLU → **dropout** | 4096 | 37,752,832 | 37,752,832 |
| fc2 | 4096 → 4096 → ReLU → **dropout** | 4096 | 16,781,312 | 16,781,312 |
| fc3 | 4096 → 1000 (logits) | 1000 | 4,097,000 | 4,097,000 |
| **Total** | | | **60,965,224** | **725,066,088** |

### The parameter/compute inversion

| | Params | Share | Mult-adds | Share |
|---|---:|---:|---:|---:|
| Conv stack (conv1–5) | 2,334,080 | 3.8% | 666,434,944 | 91.9% |
| FC stack (fc1–3) | 58,631,144 | 96.2% | 58,631,144 | 8.1% |

The conv stack does nearly all the compute and holds almost none of the
weights. The FC stack is the reverse.

The docstring in `alexnet.py` gives 665,784,864 / 58,621,952 mult-adds. Those
figures count weights only. The table above includes biases, matching
`torchinfo`. The difference is the bias term and does not affect the shares.

## Verification

| | This implementation | Paper |
|---|---:|---:|
| Total parameters | **60,965,224** | "60 million parameters" (abstract) |
| Mult-adds | **725.07 M** | not stated |

The paper states only the rounded total. The per-layer figures labelled
`paper` in `alexnet.py`'s self-check are **derived** from the layer
specification in §3.5 and Figure 2, not quoted from the paper. The one number
the paper does publish, 60 million, matches.

`torchinfo` labels its mult-adds line `(Units.MEGABYTES)`. That label is
wrong: the value is a count, 725.07 **million** multiply-adds.

## Details most implementations get wrong

**1. LRN comes *before* max-pooling, not after.** §3.5: "Response-normalization
layers follow the first and second convolutional layers. Max-pooling layers …
follow both response-normalization layers as well as the fifth convolutional
layer." The order is Conv → ReLU → LRN → MaxPool. The self-check asserts this.

**2. The LRN `alpha` must be 5e-4 in PyTorch to get the paper's 1e-4.** §3.3
uses k = 2, n = 5, α = 10⁻⁴, β = 0.75 in the formula
b = a / (k + α Σ a²)^β. PyTorch's `nn.LocalResponseNorm` computes
**α / n** · Σ a². Passing `alpha=1e-4` therefore gives an effective α of
2e-5, five times weaker than the paper. This implementation passes `alpha=5e-4`.
It is numerically checked against the paper's formula.

**3. `groups=2` on conv2, conv4 and conv5 is the two-GPU split.** §3.2: the
kernels of conv2, conv4 and conv5 connect only to maps on the same GPU. Only
conv3 (and the FC layers) see across both halves. The split came from the
GTX 580's 3 GB of memory, yet the paper reports it cut top-1 / top-5 error
by 1.7% / 1.2% compared with a one-GPU net with half as many kernels. Without grouping those three layers would
hold 2,827,136 parameters instead of 1,414,016. Most reference
implementations, including `torchvision.models.alexnet`, drop the grouping,
which is one reason their parameter counts don't match the paper's.

**4. Bias is initialised to 1 on specific layers only.** §5: "We initialized
the neuron biases in the second, fourth, and fifth convolutional layers, as
well as in the fully-connected hidden layers, with the constant 1." That
means conv2, conv4, conv5, fc1 and fc2. conv1, conv3 and the output layer
fc3 get 0. The point is to give the ReLUs positive inputs early on, so units
don't start dead. The layers are referenced by name (`self.features.conv2`,
`self.classifier.fc1`, …), not by `Sequential` index, so reordering the
layers can't silently re-target the initialisation.

**5. Dropout is only on the FC layers, and on their *outputs*.** §4.2 uses
dropout "in the first two fully-connected layers", setting "to zero the
output of each hidden neuron". fc1 and fc2 alone hold **89.4%** of the
network's parameters. Dropout goes where the parameters are, and none goes on
the conv stack, which holds 3.8%. The order is Linear → ReLU → Dropout.
The common `torchvision` order, Dropout → Linear, instead drops the 9,216
conv features (not in the paper) and never drops fc2's output (in the paper).

**6. Pooling overlaps: 3×3 windows at stride 2.** §3.4: this cut top-1 / top-5
error by 0.4% / 0.3% compared with non-overlapping 2×2 pooling.

## Caveats

- **The 224 input does not produce the paper's 55×55 map.** Figure 2 shows
  224×224×3 input, 11×11 kernels at stride 4, and a 55×55 output. Without
  padding, (224 − 11) / 4 + 1 = 54.25. This implementation uses `padding=2`
  to get 55×55. Another common fix is a 227×227 input with no padding. Both
  give identical parameter counts. The paper's own neuron count for conv1
  (253,440, §3.5) also disagrees with 96 × 55 × 55 = 290,400. This is a known
  inconsistency in the paper, not something this implementation can resolve.
- **The input normalisation is not the paper's.** §2: the only preprocessing
  was "subtracting the mean activity over the training set from each pixel".
  `train_transform` / `test_transform` use torchvision's per-channel ImageNet
  mean/std normalisation instead. Likewise, the paper resizes the shorter
  side to 256 and then center-crops 256×256 before taking random 224 crops.
  Here random crops are taken directly from the resized image.
- **PCA colour augmentation is implemented but not wired in.** `PCALighting`
  is commented out of `train_transform`. Its eigenvalues/eigenvectors are the
  widely used ImageNet values from later open-source code. The paper
  describes the method (§4.1) but does not publish the numbers.
- **There is no training loop.** `build_optimiser` gives the §5 SGD settings
  (batch 128, momentum 0.9, weight decay 5e-4, lr 0.01). The
  divide-by-10-on-plateau schedule and the training itself are not
  implemented.
