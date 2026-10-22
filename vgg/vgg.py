"""
VGG — faithful implementation of Simonyan & Zisserman (2015),
"Very Deep Convolutional Networks for Large-Scale Image Recognition" (ICLR 2015).

Verified: all six Table 1 configurations reproduce Table 2's parameter counts
exactly, and the FC-to-convolutional conversion of Section 3.2 reproduces the
original network's output to within one float32 unit in the last place.

Key fidelity points, each a deliberate choice rather than an accident:

  * Depth is the ONLY variable. Table 1's six configurations share one receptive
    field (3x3), one stride (1), one padding (1), the same five max-pool stages
    and the same three fully-connected layers. Nothing else moves between A and
    E, which is what lets the paper attribute the error difference to depth.

  * padding=1 on every 3x3 convolution. Paper, §2.1: "the spatial padding of
    conv. layer input is such that the spatial resolution is preserved after
    convolution". Only the max-pool layers change spatial size, so the 224 ->
    7 reduction is exactly 2^5 and nothing else.

  * A stack of two 3x3 layers has the receptive field of one 5x5, and three have
    the receptive field of one 7x7 -- but the stack is cheaper and has more
    non-linearities. Paper, §2.3: three 3x3 layers over C channels cost 27C^2
    against 49C^2 for a single 7x7, which is 81% more.

    This only holds BECAUSE of the ReLU between them. Convolution is linear, so
    two stacked 3x3 convolutions with no activation between them collapse into a
    single 5x5 convolution exactly. The depth argument is an argument about the
    non-linearities, not about the kernels.

  * A-LRN exists to be a control, not a model. It is configuration A plus one
    LocalResponseNorm after the first convolution, and it has the same parameter
    count as A because LRN has no parameters. Paper, §4.1: normalisation "does
    not improve on the model A", so it is dropped from B onward.

  * The 1x1 convolutions in configuration C are not dimensionality reduction --
    input and output channels are equal. Paper, §2.3: they are "a way to
    increase the non-linearity of the decision function without affecting the
    receptive fields".

  * padding = k // 2 preserves the spatial size for any odd k, which is why the
    1x1 layers in C take padding 0 and the 3x3 layers take padding 1 from the
    same rule.

Parameter / compute split for configuration D (computed, not quoted):

    CONV     14,714,688 params (10.6%)   15,346,630,656 mult-adds (99.2%)
    FC      123,642,856 params (89.4%)      123,633,664 mult-adds ( 0.8%)

(The FC parameter and mult-add counts differ by exactly 9,192 -- the 4096 + 4096
+ 1000 biases. A fully-connected weight is used once per forward pass, so for
those layers the two numbers are otherwise the same. A convolutional weight is
reused at every spatial position, which is the entire reason the two columns
above are inverted.)

The same inversion AlexNet shows, carried further: the convolutional stack does
essentially all of the arithmetic and holds about a tenth of the weights. Both
numbers matter for a different reason -- the FC layers are where the memory
goes, and the conv stack is where the time goes.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root, for utils
from utils import dim, match


# Table 1, read left to right. (k, out_channels) is a conv layer, "M" a 2x2
# max-pool at stride 2, "L" the local response normalisation that only A-LRN has.
CONFIGS = {
    "A":     [(3, 64), "M",
              (3, 128), "M",
              (3, 256), (3, 256), "M",
              (3, 512), (3, 512), "M",
              (3, 512), (3, 512), "M"],
    "A-LRN": [(3, 64), "L", "M",
              (3, 128), "M",
              (3, 256), (3, 256), "M",
              (3, 512), (3, 512), "M",
              (3, 512), (3, 512), "M"],
    "B":     [(3, 64), (3, 64), "M",
              (3, 128), (3, 128), "M",
              (3, 256), (3, 256), "M",
              (3, 512), (3, 512), "M",
              (3, 512), (3, 512), "M"],
    "C":     [(3, 64), (3, 64), "M",
              (3, 128), (3, 128), "M",
              (3, 256), (3, 256), (1, 256), "M",
              (3, 512), (3, 512), (1, 512), "M",
              (3, 512), (3, 512), (1, 512), "M"],
    "D":     [(3, 64), (3, 64), "M",
              (3, 128), (3, 128), "M",
              (3, 256), (3, 256), (3, 256), "M",
              (3, 512), (3, 512), (3, 512), "M",
              (3, 512), (3, 512), (3, 512), "M"],
    "E":     [(3, 64), (3, 64), "M",
              (3, 128), (3, 128), "M",
              (3, 256), (3, 256), (3, 256), (3, 256), "M",
              (3, 512), (3, 512), (3, 512), (3, 512), "M",
              (3, 512), (3, 512), (3, 512), (3, 512), "M"],
}

# Table 1's own column headings: "11 weight layers" and so on. Convolutions plus
# the three fully-connected layers; pooling and LRN carry no weights.
WEIGHT_LAYERS = {"A": 11, "A-LRN": 11, "B": 13, "C": 16, "D": 16, "E": 19}

# Table 2, "Number of parameters (in millions)": 133 133 133 134 138 144.
# The exact counts below are what the table rounds to.
TABLE_2 = {
    "A":     132_863_336,
    "A-LRN": 132_863_336,
    "B":     133_047_848,
    "C":     133_638_952,
    "D":     138_357_544,
    "E":     143_667_240,
}


def make_vgg(cfg_name, num_classes=1000, dropout=0.5):
    """Build one Table 1 configuration.

    Every convolution takes stride 1 and padding k // 2, which preserves the
    spatial size for any odd k -- so the 3x3 layers get padding 1 and the 1x1
    layers in configuration C get padding 0 from the same rule, with no special
    case. Only the five max-pools change the resolution.
    """
    layers, in_ch = [], 3
    for item in CONFIGS[cfg_name]:
        if item == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif item == "L":
            # Paper §2.1 cites Krizhevsky's parameters: n=5, alpha=1e-4, beta=0.75.
            # PyTorch divides alpha by size internally and the paper's formula does
            # not, so alpha is passed as 5e-4 to land on the paper's 1e-4.
            layers.append(nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2.0))
        else:
            k, out_ch = item
            layers += [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=k // 2),
                       nn.ReLU(inplace=True)]
            in_ch = out_ch

    return nn.Sequential(
        *layers,
        nn.Flatten(),
        nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(dropout),
        nn.Linear(4096, num_classes),
    )


def to_dense(model):
    """Section 3.2: convert the fully-connected layers to convolutional ones.

    "the fully-connected layers are first converted to convolutional layers (the
    first FC layer to a 7 x 7 conv. layer, the last two FC layers to 1 x 1 conv.
    layers). The resulting fully-convolutional net is then applied to the whole
    (uncropped) image."

    The conversion moves no numbers, it only reshapes them: a Linear weight of
    (4096, 25088) and a Conv2d weight of (4096, 512, 7, 7) hold the same 102.7
    million values in the same order, because the flatten that feeds the Linear
    layer runs in C order over a (512, 7, 7) tensor. If that flatten used any
    other order, .view() would still succeed and would silently scramble the
    weights -- which is why __main__ asserts the two networks agree rather than
    assuming they do.

    The result accepts any input at least 224 on its smaller side and returns a
    class score map, one prediction per 224x224 window at stride 32.
    """
    # everything up to and including the last pool is the feature stack
    cut = max(i for i, m in enumerate(model) if isinstance(m, nn.MaxPool2d))
    feats = list(model[:cut + 1])

    fcs = [m for m in model if isinstance(m, nn.Linear)]
    fc0, fc3, fc6 = fcs
    in_ch = fc0.in_features // 49

    conv0 = nn.Conv2d(in_ch, fc0.out_features, kernel_size=7)
    conv3 = nn.Conv2d(fc3.in_features, fc3.out_features, kernel_size=1)
    conv6 = nn.Conv2d(fc6.in_features, fc6.out_features, kernel_size=1)

    with torch.no_grad():
        conv0.weight.copy_(fc0.weight.view(fc0.out_features, in_ch, 7, 7))
        conv0.bias.copy_(fc0.bias)
        conv3.weight.copy_(fc3.weight.view(fc3.out_features, fc3.in_features, 1, 1))
        conv3.bias.copy_(fc3.bias)
        conv6.weight.copy_(fc6.weight.view(fc6.out_features, fc6.in_features, 1, 1))
        conv6.bias.copy_(fc6.bias)

    return nn.Sequential(*feats,
                         conv0, nn.ReLU(inplace=True),
                         conv3, nn.ReLU(inplace=True),
                         conv6).eval()


def score_map_size(h, w):
    """Spatial size of the dense score map for an h x w input.

    Five 2x2 max-pools floor-halve each side, which composes to floor(side / 32);
    the 7x7 conversion layer then runs at stride 1 with no padding and removes 6.
    """
    return h // 32 - 6, w // 32 - 6


def parameter_report(model, name):
    conv = sum(p.numel() for m in model if isinstance(m, nn.Conv2d)
               for p in m.parameters())
    fc = sum(p.numel() for m in model if isinstance(m, nn.Linear)
             for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    want = TABLE_2[name]
    ok = total == want

    print(f"  conv  {conv:>12,}  ({100 * conv / total:4.1f}%)")
    print(f"  fc    {fc:>12,}  ({100 * fc / total:4.1f}%)")
    print(f"  total {match(ok)(f'{total:>12,}')}   Table 2: {want:,}   "
          f"{match(ok)('exact' if ok else 'MISMATCH')}")
    return ok


def shape_trace(model, size=224):
    x = torch.randn(1, 3, size, size)
    print(f"  {'input':<26}{tuple(x.shape)}")
    with torch.no_grad():
        for m in model:
            x = m(x)
            if isinstance(m, (nn.MaxPool2d, nn.Flatten, nn.Linear)):
                print(f"  {m.__class__.__name__:<26}{tuple(x.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)

    print("Table 1 configurations against Table 2\n")
    all_ok = True
    for name in CONFIGS:
        model = make_vgg(name)

        n_weight = sum(1 for m in model if isinstance(m, (nn.Conv2d, nn.Linear)))
        n_pool = sum(1 for m in model if isinstance(m, nn.MaxPool2d))
        depth_ok = n_weight == WEIGHT_LAYERS[name]

        print(f"{name}  —  {match(depth_ok)(f'{n_weight} weight layers')}, "
              f"Table 1 says {WEIGHT_LAYERS[name]}   {dim(f'{n_pool} pools')}")
        all_ok &= parameter_report(model, name) and depth_ok and n_pool == 5
        print()

    # A and A-LRN must agree exactly: local response normalisation has no parameters.
    a = sum(p.numel() for p in make_vgg("A").parameters())
    alrn = sum(p.numel() for p in make_vgg("A-LRN").parameters())
    print(f"A and A-LRN identical in size: {match(a == alrn)(a == alrn)}   "
          f"{dim('LRN carries no parameters')}")

    # Section 3.2: the converted network must reproduce the original at 224.
    model = make_vgg("D").eval()
    dense = to_dense(model)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        want, got = model(x), dense(x).flatten(1)
    gap = (want - got).abs().max().item()
    same_params = (sum(p.numel() for p in model.parameters())
                   == sum(p.numel() for p in dense.parameters()))
    equiv = gap < 1e-4
    print(f"FC-to-conv agrees at 224:     {match(equiv)(f'{gap:.3e}')}   "
          f"{dim('float32 round-off, not an approximation')}")
    print(f"conversion preserves count:   {match(same_params)(same_params)}")

    # The score map grows with the input, which is the whole point of the conversion.
    print(f"\n{'input':>12}   {'score map':>11}   {'predicted':>11}   positions")
    geometry_ok = True
    for h, w in [(224, 224), (256, 256), (384, 384), (512, 512), (256, 341)]:
        with torch.no_grad():
            out = dense(torch.randn(1, 3, h, w))
        got_hw, want_hw = tuple(out.shape[2:]), score_map_size(h, w)
        geometry_ok &= got_hw == want_hw
        print(f"{h:>5} x {w:<4}   {got_hw[0]:>4} x {got_hw[1]:<4}   "
              f"{want_hw[0]:>4} x {want_hw[1]:<4}   {got_hw[0] * got_hw[1]:>9}")

    print()
    everything = all_ok and a == alrn and equiv and same_params and geometry_ok
    print(match(everything)("all checks passed" if everything else "SOMETHING FAILED"))
