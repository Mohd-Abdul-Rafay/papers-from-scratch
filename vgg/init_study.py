"""
Section 3.1 — why the deep configurations needed pre-initialisation.

The paper gives the problem two sentences and the fix a paragraph:

    "The initialisation of the network weights is important, since bad
    initialisation can stall learning due to the instability of gradient in deep
    nets. To circumvent this problem, we began with training the configuration
    A ... Then, when training deeper architectures, we initialised the first
    four convolutional layers and the last three fully-connected layers with the
    layers of net A."

Read once, it looks like an engineering workaround. Run it and it is the
difference between a network that learns and one that does not.

This script runs configuration E on Imagenette — 10 classes, ~9.5k training
images — across four initialisations and five learning rates, then runs the
paper's transplant. It is a small proxy and the README says so: ten classes are
not a thousand and five epochs are not seventy-four. What it can settle is which
initialisations train a sixteen-layer rectified network AT ALL, which is the
question section 3.1 is about.

One thing here could not be reproduced. The paper trains its donor A with its own
random initialisation. That initialisation does not train configuration A here
either, at any rate tried, so the paper's donor does not exist to be copied from.
The donor used below is Kaiming-trained, which makes this a test of whether a
trained A rescues E — not a test of the paper's method end to end. Saying so is
the point; quietly substituting a donor and reporting the result as the paper's
would not be.

    python init_study.py --data /path/to/imagenette2
    python init_study.py --data ... --only transplant

Imagenette: https://github.com/fastai/imagenette
"""

import argparse
import json
import math
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vgg import make_vgg

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

LEARNING_RATES = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

# The paper subtracts the training-set mean and does not divide by the standard
# deviation (section 2.1). These runs train from scratch, so they follow it —
# unlike dense_eval.py, which uses torchvision's pretrained weights and must use
# torchvision's normalisation instead. Mixing the two is silent and costly.
MEAN = [0.485, 0.456, 0.406]
NORMALISE = T.Normalize(mean=MEAN, std=[1.0, 1.0, 1.0])


def initialise(model, how):
    """Four initialisations, three of them the paper's contemporaries."""
    for m in model.modules():
        if not isinstance(m, (nn.Conv2d, nn.Linear)):
            continue
        if how == "paper-0.1":
            # the paper's "zero-mean Gaussian with 1e-2 variance", read literally:
            # variance 1e-2 is standard deviation 0.1
            nn.init.normal_(m.weight, 0.0, 0.1)
        elif how == "paper-0.01":
            # and read as the authors almost certainly meant it
            nn.init.normal_(m.weight, 0.0, 0.01)
        elif how == "glorot":
            nn.init.xavier_normal_(m.weight)
        elif how == "kaiming":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        else:
            raise ValueError(how)
        nn.init.zeros_(m.bias)
    return model


def loaders(root, batch=32, workers=8):
    train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                          T.ToTensor(), NORMALISE])
    val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORMALISE])
    tr = ImageFolder(os.path.join(root, "train"), train_tf)
    va = ImageFolder(os.path.join(root, "val"), val_tf)
    return (DataLoader(tr, batch, shuffle=True, num_workers=workers, pin_memory=True),
            DataLoader(va, batch, num_workers=workers, pin_memory=True),
            len(tr.classes))


def train(model, tr, va, lr, epochs, device):
    """Returns validation accuracy, or None if the run diverged.

    A diverged run is reported as None rather than as some number near chance.
    They are different failures: one produced NaN and stopped being a model, the
    other trained and learned nothing.
    """
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                return None
            loss.backward()
            opt.step()
    model.eval()
    right = n = 0
    with torch.no_grad():
        for x, y in va:
            x, y = x.to(device), y.to(device)
            right += int((model(x).argmax(1) == y).sum())
            n += y.numel()
    return 100 * right / n


def grid(tr, va, classes, epochs, device):
    print("\nConfiguration E, validation accuracy after "
          f"{epochs} epochs. Chance is {100 / classes:.1f}%.\n")
    print(f"  {'initialisation':<22}" + "".join(f"{lr:>9.0e}" for lr in LEARNING_RATES))
    rows = []
    for how, note in [("paper-0.1",  "the paper's variance 1e-2 read literally"),
                      ("paper-0.01", "the paper's std 0.01"),
                      ("glorot",     "half Kaiming's variance per layer"),
                      ("kaiming",    "variance 2 / fan_in")]:
        acc = []
        for lr in LEARNING_RATES:
            torch.manual_seed(0)
            a = train(initialise(make_vgg("E", num_classes=classes), how),
                      tr, va, lr, epochs, device)
            acc.append(None if a is None else round(a, 1))
        rows.append({"init": how, "note": note, "acc": acc,
                     "nan": [a is None for a in acc]})
        cells = "".join("      NaN" if a is None else f"{a:9.1f}" for a in acc)
        print(f"  {how:<22}{cells}")

    flat = [a for r in rows for a in r["acc"]]
    trains = sum(1 for a in flat if a is not None and a > 2 * 100 / classes)
    nan = sum(1 for a in flat if a is None)
    print(f"\n  {trains} cells train, {len(flat) - trains - nan} sit near chance, "
          f"{nan} diverged.")
    return {"config": "E", "epochs": epochs, "chance_pct": 100 / classes,
            "learning_rates": LEARNING_RATES, "rows": rows,
            "summary": {"cells_that_train": trains,
                        "cells_near_chance": len(flat) - trains - nan,
                        "cells_diverged": nan}}


def transplant(tr, va, classes, epochs, device, lr=1e-3):
    """Section 3.1: train A, then copy its layers into E.

    The paper specifies the transplant BY LAYER, not by position — "the first
    four convolutional layers" of A, which are A's conv1 through conv4. In E
    those same layers sit at different positions, because E has two convolutions
    where A has one in the early blocks. Copying A's first four convolutions into
    E's first four positions copies the right weights into the wrong shapes.
    """
    print("\nPre-initialisation (section 3.1)\n")
    torch.manual_seed(0)
    donor = initialise(make_vgg("A", num_classes=classes), "kaiming")
    acc_a = train(donor, tr, va, lr, epochs, device)
    print(f"  donor A, Kaiming, lr {lr:g}: {acc_a:.1f}%")

    torch.manual_seed(0)
    deep = initialise(make_vgg("E", num_classes=classes), "glorot")
    a_conv = [m for m in donor if isinstance(m, nn.Conv2d)]
    e_conv = [m for m in deep if isinstance(m, nn.Conv2d)]
    a_fc = [m for m in donor if isinstance(m, nn.Linear)]
    e_fc = [m for m in deep if isinstance(m, nn.Linear)]

    copied = []
    with torch.no_grad():
        # the first four of A's convolutions, matched to E's by SHAPE, not index
        for src in a_conv[:4]:
            for i, dst in enumerate(e_conv):
                if dst.weight.shape == src.weight.shape and i not in copied:
                    dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
                    copied.append(i)
                    break
        for src, dst in zip(a_fc, e_fc):          # and the last three FC layers
            dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
    print(f"  copied {len(copied)} convolutions (E positions {copied}) and "
          f"{len(a_fc)} fully-connected layers")

    acc_e = train(deep, tr, va, lr, epochs, device)
    torch.manual_seed(0)
    acc_plain = train(initialise(make_vgg("E", num_classes=classes), "glorot"),
                      tr, va, lr, epochs, device)
    print(f"  E + pre-init, Glorot elsewhere: {acc_e:.1f}%")
    print(f"  E alone, Glorot:                "
          f"{'diverged' if acc_plain is None else f'{acc_plain:.1f}%'}")
    print("\n  The paper's own donor could not be reproduced: its random initialisation")
    print("  does not train A here either. This donor is Kaiming-trained, so the result")
    print("  says a trained A rescues E — not that the paper's method does.")

    return {"paper_section": "3.1",
            "donor": {"config": "A", "init": "kaiming", "val_acc_pct": acc_a},
            "transplant": {"config": "E",
                           "layers_initialised_from_donor": len(copied) + len(a_fc),
                           "remaining_layers_init": "glorot", "lr": lr,
                           "val_acc_pct": acc_e},
            "control_E_alone_glorot": acc_plain,
            "could_not_be_reproduced": {
                "what": "the paper trains its donor A with its own random initialisation",
                "why": "that initialisation does not train A here either, at any rate tried",
                "consequence": "a test of whether a trained A rescues E, not of the "
                               "paper's method end to end"}}


def deficits():
    """The model behind the signal-deficit figure, stated so it can be checked.

    Glorot gives variance 2 / (fan_in + fan_out); Kaiming gives 2 / fan_in. Where
    fan_in equals fan_out that is exactly half, so the shortfall in standard
    deviation is 1/sqrt(2) per layer and compounds as a power of the depth.

    These are MODELLED. One of them has been compared with a measurement — for E
    alone under Glorot the model gives 256x and the measured activation RMS ratio
    at the last convolution was 1,571x, six times larger. Read the axis as an
    ordering, not a physical quantity.
    """
    out = []
    for case, init, layers in [("A alone", "glorot", 8), ("E + pre-init", "glorot", 12),
                               ("E alone", "glorot", 16), ("E + pre-init", "paper init", 12),
                               ("E alone", "paper init", 16)]:
        ratio = math.sqrt(2) if init == "glorot" else 0.02 / 0.01 * 1.3
        out.append({"case": case, "init": init, "layers": layers,
                    "deficit": round(ratio ** layers)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", required=True, help="imagenette2 root (train/ and val/)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only", choices=["grid", "transplant"], nargs="*")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: no CUDA. The grid is 20 training runs and will take days.")
    tr, va, classes = loaders(args.data, args.batch)
    print(f"{len(tr.dataset):,} train, {len(va.dataset):,} val, {classes} classes "
          f"· {args.epochs} epochs · {device}")

    path = os.path.join(RESULTS, "init_study.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["_provenance"] = {
        "produced_by": "vgg/init_study.py",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "data": os.path.abspath(args.data),
        "epochs": args.epochs, "batch": args.batch,
        "normalisation": "mean subtracted only, as the paper specifies",
        "note": "Imagenette is not ILSVRC and five epochs are not seventy-four. "
                "This settles which initialisations train a deep rectified network "
                "at all, not what accuracy VGG reaches.",
    }
    want = args.only or ["grid", "transplant"]
    if "grid" in want:
        out["heatmap"] = grid(tr, va, classes, args.epochs, device)
    if "transplant" in want:
        out["pre_initialisation"] = transplant(tr, va, classes, args.epochs, device)
    out.setdefault("signal_deficit", {})["bars_modelled"] = deficits()

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
