"""
The CIFAR architecture of section 4.2, built from the paper's own description
and checked against Table 6.

Section 4.2, in full:
  "The first layer is 3x3 convolutions. Then we use a stack of 6n layers with
   3x3 convolutions on the feature maps of sizes {32, 16, 8} respectively,
   with 2n layers for each feature map size. The numbers of filters are
   {16, 32, 64} respectively. The subsampling is performed by convolutions
   with a stride of 2. The network ends with a global average pooling, a
   10-way fully-connected layer, and softmax. There are totally 6n+2 stacked
   weighted layers."
  "When shortcut connections are used, they are connected to the pairs of
   3x3 layers (totally 3n shortcuts). On this dataset we use identity
   shortcuts in all cases (i.e., option A), so our residual models have
   exactly the same depth, width, and number of parameters as the plain
   counterparts."

Run it on its own to see where 6n+2 comes from and that the counts match:
    python cifar_resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn

PAPER_TABLE_6 = {20: 0.27e6, 32: 0.46e6, 44: 0.66e6, 56: 0.85e6, 110: 1.7e6, 1202: 19.4e6}


class ShortcutA(nn.Module):
    """Option A: subsample by taking every other pixel, pad the new channels
    with zeros. No parameters, which is the whole point -- it keeps the
    residual net's parameter count identical to the plain net's."""

    def __init__(self, cin, cout, stride):
        super().__init__()
        self.stride = stride
        self.extra = cout - cin

    def forward(self, x):
        if self.stride == 2:
            x = x[:, :, ::2, ::2]
        if self.extra:
            x = Fn.pad(x, (0, 0, 0, 0, 0, self.extra))
        return x


class Block(nn.Module):
    """Two 3x3 weight layers. residual=False gives the plain counterpart."""

    def __init__(self, cin, cout, stride, residual):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.residual = residual
        if residual:
            self.short = (ShortcutA(cin, cout, stride)
                          if (stride != 1 or cin != cout) else nn.Identity())

    def forward(self, x):
        out = Fn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.short(x)
        return Fn.relu(out)


class CifarNet(nn.Module):
    def __init__(self, n, residual, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)  # first of the +2
        self.bn1 = nn.BatchNorm2d(16)

        blocks, cin = [], 16
        for stage, cout in enumerate([16, 32, 64]):           # feature maps 32, 16, 8
            for b in range(n):                                 # 2n weight layers per size
                stride = 2 if (stage > 0 and b == 0) else 1    # subsample by stride-2 conv
                blocks.append(Block(cin, cout, stride, residual))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(64, num_classes)                   # second of the +2

        # "we adopt the weight initialization in [12]" -- He et al. 2015
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = Fn.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = Fn.adaptive_avg_pool2d(x, 1).flatten(1)            # global average pooling
        return self.fc(x)


def build(name):
    """'plain20' / 'resnet56' -> (model, depth)."""
    if name.startswith("resnet"):
        kind, depth = "resnet", int(name[6:])
    elif name.startswith("plain"):
        kind, depth = "plain", int(name[5:])
    else:
        raise ValueError(f"model name must start with 'plain' or 'resnet', got {name!r}")
    n = (depth - 2) // 6
    assert 6 * n + 2 == depth, f"{depth} is not of the form 6n+2"
    return CifarNet(n, residual=(kind == "resnet")), depth


def weight_layers(model):
    """The paper's own counting rule: convolutions and fully-connected layers
    only. BatchNorm, pooling and ReLU are never counted."""
    return sum(1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))


def shortcuts(model):
    return sum(1 for m in model.modules() if isinstance(m, Block) and m.residual)


if __name__ == "__main__":
    print("=" * 74)
    print("Where 6n+2 comes from")
    print("=" * 74)
    net = CifarNet(n=3, residual=True)
    print("  n = 3")
    print("    first 3x3 conv                        :  1")
    print(f"    stack: 3 sizes x 3 blocks x 2 convs   : {6*3}   (= 6n)")
    print("    final fully-connected layer           :  1")
    print(f"    {'-' * 42}")
    print(f"    total weight layers                   : {weight_layers(net)}   (= 6n+2)")
    print("    BatchNorm layers, counted as none     : "
          f"{sum(1 for m in net.modules() if isinstance(m, nn.BatchNorm2d))}")
    print(f"    shortcuts (paper says 3n)             : {shortcuts(net)}")

    print("\n" + "=" * 74)
    print("Against Table 6")
    print("=" * 74)
    print(f"  {'depth':>5}  {'n':>3}  {'layers':>6}  {'residual params':>15}  "
          f"{'plain params':>13}  {'paper':>7}  {'match':>5}")
    print(f"  {'-'*5}  {'-'*3}  {'-'*6}  {'-'*15}  {'-'*13}  {'-'*7}  {'-'*5}")
    all_ok = True
    for n in (3, 5, 7, 9, 18):
        depth = 6 * n + 2
        r = CifarNet(n, residual=True)
        p = CifarNet(n, residual=False)
        rp = sum(q.numel() for q in r.parameters())
        pp = sum(q.numel() for q in p.parameters())
        paper = PAPER_TABLE_6[depth]
        places = 2 if depth < 110 else 1
        hit = round(rp / 1e6, places) == round(paper / 1e6, places)
        all_ok &= hit and (rp == pp)
        print(f"  {depth:>5}  {n:>3}  {weight_layers(r):>6}  {rp:>15,}  {pp:>13,}  "
              f"{paper/1e6:>6.2f}M  {'yes' if hit else 'NO':>5}")

    print()
    print("  plain and residual have identical parameter counts :",
          all(sum(q.numel() for q in CifarNet(n, True).parameters()) ==
              sum(q.numel() for q in CifarNet(n, False).parameters())
              for n in (3, 5, 7, 9, 18)))
    print("  -> which is what option A buys: the only difference is the shortcut.")

    print("\n" + "=" * 74)
    print("Shapes through ResNet-20")
    print("=" * 74)
    net = CifarNet(3, residual=True).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        h = Fn.relu(net.bn1(net.conv1(x)))
        print(f"  after the first conv        : {tuple(h.shape)}")
        for i, blk in enumerate(net.blocks):
            h = blk(h)
            tag = ("   <- stride 2, shortcut pads channels"
                   if isinstance(getattr(blk, "short", None), ShortcutA) else "")
            print(f"  after block {i+1:<2}              : {tuple(h.shape)}{tag}")
        g = Fn.adaptive_avg_pool2d(h, 1).flatten(1)
        print(f"  after global average pooling: {tuple(g.shape)}")
        print(f"  after the fully-connected   : {tuple(net.fc(g).shape)}")

    print()
    raise SystemExit(0 if all_ok else 1)
