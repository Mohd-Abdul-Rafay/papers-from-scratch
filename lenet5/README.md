# LeNet-5

**Paper:** Y. LeCun, L. Bottou, Y. Bengio, P. Haffner.
*Gradient-Based Learning Applied to Document Recognition.*
Proceedings of the IEEE, 86(11):2278–2324, 1998.
[DOI 10.1109/5.726791](https://doi.org/10.1109/5.726791) ·
[author's publication page](http://yann.lecun.com/exdb/publis/index.html)
(not on arXiv)

```bash
python lenet5/lenet5.py     # parameter report, shape check, loss check, prototype check
```

## Architecture

Input is a single 32×32 channel. "Connections" are the paper's figures (§II.B),
except where marked.

| Layer | Operation | Output shape | Params | Connections |
|---|---|---|---:|---:|
| Input | MNIST 28×28, padded by 2 | 1 × 32 × 32 | — | — |
| C1 | 5×5 conv, 6 maps | 6 × 28 × 28 | 156 | 122,304 |
| S2 | 2×2 sum × trainable coeff + bias, squash | 6 × 14 × 14 | 12 | 5,880 |
| C3 | 5×5 conv, 16 maps, partial connections (Table I) | 16 × 10 × 10 | 1,516 | 151,600 |
| S4 | 2×2 sum × trainable coeff + bias, squash | 16 × 5 × 5 | 32 | 2,000 |
| C5 | 5×5 conv, 120 maps | 120 × 1 × 1 | 48,120 | 48,120 |
| F6 | fully connected | 84 | 10,164 | 10,164 ¹ |
| Output | Euclidean RBF, 10 fixed prototypes | 10 | 0 | 840 ² |
| **Total** | | | **60,000** | **340,908** |

¹ The paper gives F6's parameter count only. It is fully connected, so connections = parameters.
² The paper does not state this figure: 84 inputs × 10 classes. See the reconciliation below.

## Verification

### Parameters: all six layers match the paper exactly

| Layer | This implementation | Paper (§II.B) |
|---|---:|---:|
| C1 | 156 | 156 |
| S2 | 12 | 12 |
| C3 | 1,516 | 1,516 |
| S4 | 32 | 32 |
| C5 | 48,120 | 48,120 |
| F6 | 10,164 | 10,164 |
| **Total** | **60,000** | **60,000** (§II.A: "only 60,000 trainable free parameters") |

`torchinfo` reports `Total params: 60,000`, all trainable. The RBF prototypes are a
registered buffer, not a parameter, so they are correctly excluded.

### Connections: reconciling the paper's total

§II.A states that the network "contains 340,908 connections". §II.B gives the
per-layer figures, and these do **not** add up to 340,908 by themselves:

```
  C1   122,304   stated
  S2     5,880   stated
  C3   151,600   stated   (1,516 params × 100 output positions, from Table I)
  S4     2,000   stated
  C5    48,120   stated
  F6    10,164   stated as parameters; fully connected, so also connections
  ─────────────
       340,068   sum of the per-layer figures
  RBF      840   not stated: 84 inputs × 10 output units
  ─────────────
       340,908   = the paper's stated total, exactly
```

So the paper's total includes the 840 connections of the output RBF layer,
although §II.B never lists that layer's count. With those included, every
figure in the paper is consistent with every other.

### Connections: what `torchinfo` sees

`torchinfo` reports **0.33 M mult-adds** (332,188 exactly). That is
340,908 − 5,880 − 2,000 − 840. Its count covers only the built-in `Conv2d` and
`Linear` modules. It cannot see inside the custom S2, S4 and RBF layers, so they
contribute nothing. For C1, C3, C5 and F6 its numbers match the paper's
connection counts exactly, since it counts bias terms as one connection each,
as the paper does.

### Full torchinfo output

<details>
<summary>Full torchinfo output (1 × 1 × 32 × 32 input)</summary>

```
===================================================================================================================
Layer (type (var_name))                  Output Shape              Param #                   Mult-Adds
===================================================================================================================
LeNet5 (LeNet5)                          [1, 10]                   --                        --
├─Conv2d (c1)                            [1, 6, 28, 28]            156                       122,304
├─LeNetActivation (c1_activation)        [1, 6, 28, 28]            --                        --
├─SubsamplingLayer (s2)                  [1, 6, 14, 14]            12                        --
│    └─LeNetActivation (activation)      [1, 6, 14, 14]            --                        --
├─C3Layer (c3)                           [1, 16, 10, 10]           --                        --
│    └─ModuleList (convs)                --                        --                        --
│    │    └─Conv2d (0)                   [1, 1, 10, 10]            76                        7,600
│    │    └─Conv2d (1)                   [1, 1, 10, 10]            76                        7,600
│    │    └─Conv2d (2)                   [1, 1, 10, 10]            76                        7,600
│    │    └─Conv2d (3)                   [1, 1, 10, 10]            76                        7,600
│    │    └─Conv2d (4)                   [1, 1, 10, 10]            76                        7,600
│    │    └─Conv2d (5)                   [1, 1, 10, 10]            76                        7,600
│    │    └─Conv2d (6)                   [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (7)                   [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (8)                   [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (9)                   [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (10)                  [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (11)                  [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (12)                  [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (13)                  [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (14)                  [1, 1, 10, 10]            101                       10,100
│    │    └─Conv2d (15)                  [1, 1, 10, 10]            151                       15,100
│    └─LeNetActivation (activation)      [1, 16, 10, 10]           --                        --
├─SubsamplingLayer (s4)                  [1, 16, 5, 5]             32                        --
│    └─LeNetActivation (activation)      [1, 16, 5, 5]             --                        --
├─Conv2d (c5)                            [1, 120, 1, 1]            48,120                    48,120
├─LeNetActivation (c5_activation)        [1, 120, 1, 1]            --                        --
├─Linear (f6)                            [1, 84]                   10,164                    10,164
├─LeNetActivation (f6_activation)        [1, 84]                   --                        --
├─RBFLayer (rbf)                         [1, 10]                   --                        --
===================================================================================================================
Total params: 60,000
Trainable params: 60,000
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 0.33
===================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.05
Params size (MB): 0.24
Estimated Total Size (MB): 0.30
===================================================================================================================
```

The custom S2, S4 and RBF layers show no mult-adds because torchinfo cannot see inside them. The `(Units.MEGABYTES)` label is torchinfo's mistake: the value is 0.33 **million** mult-adds.

</details>

## Details most implementations get wrong

**1. The activation is a scaled tanh: `1.7159 · tanh(2/3 · x)`, not `tanh` or ReLU.**
(Appendix A.) The constants are chosen so that f(±1) = ±1. That places the ±1
targets of the RBF prototypes at the activation's points of maximum
curvature, where gradients are healthy, rather than in the saturated tails.

**2. S2 and S4 are not average pooling.** Each subsampling unit *sums* its 2×2
inputs, multiplies by a **trainable coefficient**, adds a **trainable bias**, and
applies the squashing function. That is one coefficient and one bias per map,
which is why S2 has 12 parameters and S4 has 32. Parameter-free `AvgPool2d`
gets both counts wrong (0) and drops the nonlinearity. The implementation
initialises the coefficient to 0.25. A coefficient of 1.0 would make four
summed activations of up to 1.7159 push the tanh straight into saturation.

**3. C3 is not fully connected to S2.** Following Table I, the first six C3
maps each see 3 contiguous S2 maps, the next six see 4 contiguous maps, three
see 4 non-contiguous maps, and one sees all six. This gives 1,516 parameters
instead of 2,416, and the paper's stated purpose is to break symmetry: maps
given different inputs are forced to learn different features. The table in
`C3_CONNECTIONS` matches Table I map for map.

**4. The RBF prototypes are frozen.** The output layer computes squared
Euclidean distance to ten fixed ±1 vectors. They are registered as a buffer,
not a parameter. If they were trainable, the simple MSE criterion would admit
a collapsed solution where all prototypes coincide and the network ignores its
input. `map_loss` implements the paper's discriminative MAP criterion (eq. 9),
which adds a competitive term against that collapse.

**5. Prediction is `argmin`, not `argmax`.** The network outputs *distances*,
so smaller means more likely. `argmax` on LeNet-5's output returns the
**least** likely class. The distances also give rejection for free:
`reject_aware_predict` refuses inputs that are far from every prototype.

**6. Input is normalised to [−0.1, 1.175], not [0, 1] or mean/std.** (§II.B.)
Background maps to −0.1 and ink to 1.175, which makes the input mean roughly 0
and variance roughly 1. MNIST's 28×28 digits are padded to 32×32 first, so
that stroke endpoints near the edge can sit at the centre of the receptive
fields of the highest-level feature detectors.

## Caveats

- **The digit RBF bitmaps are not from the paper.** The paper describes the
  prototypes as stylised 7×12 bitmaps of each character class, with entries
  ±1. It does **not** publish the bitmaps for the digit-only network. Figure 3
  shows the prototypes for the full ASCII set, not the ten digits used here.
  The bitmaps in `DIGIT_BITMAPS` are drawn by hand to satisfy the property the
  paper describes: visually confusable characters get similar codes, so
  ambiguity survives to a postprocessor. They are **not copied** from the
  paper. The self-check prints the closest prototype pairs (0–8, 0–9, 0–6,
  6–8) as a sanity check of that property.
- **The subsampling coefficient starts at a fixed 0.25.** Appendix A
  initialises weights uniformly in ±2.4 / F_in. Applied to S2 and S4
  (F_in = 4), that range would be ±0.6. This implementation uses that rule for
  the conv and linear layers but a fixed 0.25 for the subsampling coefficients.
- **There is no training loop.** The file provides the architecture, both loss
  functions and the preprocessing. The paper trained with a stochastic
  diagonal Levenberg–Marquardt method, which is not implemented here.
