"""The construction argument of section 1, measured at depth 56.

   "Consider a shallower architecture and its deeper counterpart that adds more
    layers onto it. There exists a solution by construction to the deeper model:
    the added layers are identity mapping, and the other layers are copied from
    the learned shallower model."

Set every block's convolution weights to zero -- the 'learned nothing' state --
and push a non-negative input through. That is the floor each architecture
falls back to when F learns nothing.

It also prints the ReLU count of both nets, because the shortcut is sometimes
described as adding non-linearity. It adds none: the two nets have the same
ReLUs, the same parameters and the same depth. The shortcut adds a LINEAR path,
which is the opposite, and is the point -- an identity bypass is what lets a
deep net behave like a shallow one.

    python floor.py
"""
import torch
import torch.nn as nn

from cifar_resnet import Block, CifarNet, ShortcutA


def relus(model):
    """ReLU is applied functionally here, so count it structurally: one after
    the first conv, two inside every block (after bn1, and after the sum)."""
    return 1 + 2 * sum(1 for m in model.modules() if isinstance(m, Block))


def zero_out_F(model):
    """Every block learns nothing. BatchNorm is left at its init -- gamma 1,
    beta 0, running_mean 0, running_var 1 -- so BN of zero is zero."""
    for m in model.modules():
        if isinstance(m, Block):
            nn.init.zeros_(m.conv1.weight)
            nn.init.zeros_(m.conv2.weight)
    return model.eval()     # eval() is mandatory: train() would use the batch's
                            # own statistics, and BN of an all-zero batch divides
                            # by something very close to zero.


def main():
    print(f"{'':>10}  {'params':>9}  {'ReLUs':>6}  {'weight layers':>13}")
    for residual in (False, True):
        net = CifarNet(9, residual=residual)              # 6*9 + 2 = 56
        print(f"{'resnet56' if residual else 'plain56':>10}  "
              f"{sum(p.numel() for p in net.parameters()):>9,}  {relus(net):>6}  "
              f"{sum(1 for m in net.modules() if isinstance(m, (nn.Conv2d, nn.Linear))):>13}")
    print("\n-> identical in every respect except the '+'.\n")

    torch.manual_seed(0)
    x = torch.relu(torch.randn(4, 16, 32, 32))   # non-negative, as it would be
                                                 # arriving from a ReLU
    print(f"input into the stack: mean {x.mean():.4f}  norm {x.norm():.2f}\n")
    print(f"  {'after block':>12}  {'plain56 norm':>13}  {'resnet56 norm':>14}")
    print(f"  {'-'*12}  {'-'*13}  {'-'*14}")

    nets = {k: zero_out_F(CifarNet(9, residual=(k == "resnet56")))
            for k in ("plain56", "resnet56")}
    h = {k: x.clone() for k in nets}
    with torch.no_grad():
        for i in range(27):
            for k, net in nets.items():
                h[k] = net.blocks[i](h[k])
            if i in (0, 1, 8, 9, 17, 18, 26):
                print(f"  {i+1:>12}  {h['plain56'].norm():>13.2f}  "
                      f"{h['resnet56'].norm():>14.2f}")

    print(f"\nplain56   : signal norm went to {h['plain56'].norm():.4f}")
    print(f"resnet56  : signal norm is {h['resnet56'].norm():.2f}, input was "
          f"{x.norm():.2f} -- shrunk only by the two stride-2 subsamplings")
    print("\nSame parameters, same ReLUs, same depth. The '+' is the entire")
    print("difference, and it is what stops 'learned nothing' from meaning")
    print("'destroyed the input'.")

    # How many of the 27 blocks have a parameter-free projection, i.e. option A
    net = CifarNet(9, residual=True)
    a = sum(1 for m in net.modules() if isinstance(m, ShortcutA))
    ident = sum(1 for m in net.modules() if isinstance(m, nn.Identity))
    print(f"\nshortcuts: {ident} plain identities, {a} option-A subsample-and-pad "
          f"(the two stage changes), {ident + a} in total = 3n for n = 9.")


if __name__ == "__main__":
    main()
