"""
Dense evaluation — Sections 3.2, 4.2 and 4.3 of Simonyan & Zisserman (2015),
reproduced on the ILSVRC-2012 validation set.

Section 3.2 describes a test-time procedure, not an architecture:

    "the fully-connected layers are first converted to convolutional layers (the
    first FC layer to a 7 x 7 conv. layer, the last two FC layers to 1 x 1 conv.
    layers). The resulting fully-convolutional net is then applied to the whole
    (uncropped) image. The result is a class score map with the number of
    channels equal to the number of classes, and a variable spatial resolution,
    dependent on the input image size. Finally, to obtain a fixed-size vector of
    class scores for the image, the class score map is spatially averaged
    (sum-pooled). We also augment the test set by horizontal flipping of the
    images; the soft-max class posteriors of the original and flipped images are
    averaged to obtain the final scores for the image."

Three protocols are implemented here:

  crop       resize the smaller side to Q, take the centre 224 crop, one pass.
             This is the AlexNet-era baseline the paper is arguing against.

  dense      resize the smaller side to Q, run the fully-convolutional net on
             the whole uncropped image, average the score map, soft-max.
             Equivalent to evaluating a 224 crop at every 32-pixel offset, for
             the cost of one pass, since the convolutional work is shared.

  multicrop  §4.3: "50 crops per scale (5 x 5 regular grid with 2 flips)".
             Fifty separate passes per image. The crops are 224 square whatever
             Q is, so this costs the same at every scale -- only dense gets more
             expensive as Q grows.

TWO ORDERINGS MATTER AND ARE EASY TO GET BACKWARDS:

  * the score map is averaged in LOGIT space, before the soft-max;
  * the original and flipped images are averaged in POSTERIOR space, after it.

Section 4.2 then averages posteriors across several test scales Q. For a model
trained with scale jittering the paper uses Q = {256, 384, 512}.

Measured here with torchvision's pretrained VGG-16 (configuration D) on all
50,000 ILSVRC-2012 validation images, Q = {256, 384, 512}:

    protocol             top-1      top-5        paper, Table 5
    dense               25.242%     7.764%       24.8 / 7.5
    multi-crop          25.100%     7.676%       24.6 / 7.5
    multi-crop & dense  24.812%     7.420%       24.4 / 7.2

The ordering reproduces, and so do the gaps: 0.430 against the paper's 0.4 for
dense to the combination. The ~0.45 point offset is the weights, not the
protocol -- these are torchvision's, trained with RandomResizedCrop rather than
the paper's scale jittering.

Two findings that are not in the paper:

  * The offset is nearly constant across protocols (0.44 / 0.50 / 0.41), so the
    benefit of the evaluation protocol is close to independent of how good the
    weights are.

  * Averaging over Q = {256, 384, 512} is worth about 1.1 points of top-1, while
    choosing multi-crop over dense is worth 0.14 and is not statistically
    significant. The scale lever is roughly three times the protocol lever. The
    paper gives §4.3 and a table to the smaller of the two.

Usage:
    python dense_eval.py --data /path/to/imagenet/val
    python dense_eval.py --data ... --scales 256 --protocols crop dense

The data directory is an ImageFolder tree of 1,000 wnid-named folders; the class
index is the folder's position in sorted order, which is the ImageNet-1k
convention and is asserted on startup.
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16

# torchvision's pretrained weights were trained with this normalisation. The
# paper subtracts the training-set mean only; using the paper's version with
# these weights is a silent mismatch that costs accuracy without erroring.
NORMALISE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class ImageNetVal(Dataset):
    """ImageFolder-style validation set labelled by sorted wnid position."""

    def __init__(self, root, transform):
        self.items, self.transform = [], transform
        for index, wnid in enumerate(sorted(os.listdir(root))):
            folder = os.path.join(root, wnid)
            for name in sorted(os.listdir(folder)):
                self.items.append((os.path.join(folder, name), index))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), label


def to_dense(model):
    """Convert torchvision's VGG classifier to the §3.2 convolutional head.

    The avgpool is deliberately dropped. torchvision inserts AdaptiveAvgPool2d
    ((7, 7)) so that its Linear classifier accepts any input size -- it solves
    the same problem §3.2 solves, but by discarding the spatial resolution
    instead of keeping it. Left in place it forces a 1x1 score map at every Q,
    which produces no error and quietly defeats the entire procedure.
    """
    fc0, fc3, fc6 = (model.classifier[i] for i in (0, 3, 6))
    conv0 = nn.Conv2d(512, 4096, kernel_size=7)
    conv3 = nn.Conv2d(4096, 4096, kernel_size=1)
    conv6 = nn.Conv2d(4096, 1000, kernel_size=1)
    with torch.no_grad():
        conv0.weight.copy_(fc0.weight.view(4096, 512, 7, 7))
        conv0.bias.copy_(fc0.bias)
        conv3.weight.copy_(fc3.weight.view(4096, 4096, 1, 1))
        conv3.bias.copy_(fc3.bias)
        conv6.weight.copy_(fc6.weight.view(1000, 4096, 1, 1))
        conv6.bias.copy_(fc6.bias)
    head = nn.Sequential(conv0, nn.ReLU(inplace=True),
                         conv3, nn.ReLU(inplace=True), conv6)
    return nn.Sequential(model.features, head).eval()


@torch.no_grad()
def posteriors(protocol, root, model, dense, device, Q, grid=5, cache=None):
    """Class posteriors for every image under one protocol at one test scale.

    Kept in float32 throughout. Autocast to float16 was measured on this exact
    setup: it changed the top-1 prediction of 53 images in 50,000 (0.106%) and
    was not measurably faster, because JPEG decoding, not the GPU, is the
    bottleneck. The effects being measured here are 0.1 to 0.4 points, so that
    noise is the size of the signal and buys nothing.
    """
    if cache and os.path.exists(cache):
        print(f"  {cache} exists — reusing")
        return np.load(cache)

    if protocol == "crop":
        tf = T.Compose([T.Resize(Q), T.CenterCrop(224), T.ToTensor(), NORMALISE])
        batch = 128
    else:
        tf = T.Compose([T.Resize(Q), T.ToTensor(), NORMALISE])
        batch = 1                      # aspect ratios differ, so no batching
    loader = DataLoader(ImageNetVal(root, tf), batch_size=batch,
                        num_workers=8, pin_memory=True)

    out, started = [], time.time()
    for step, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        if protocol == "multicrop":
            h, w = x.shape[2], x.shape[3]
            ys = torch.linspace(0, h - 224, grid).round().long().tolist()
            xs = torch.linspace(0, w - 224, grid).round().long().tolist()
            crops = torch.cat([x[:, :, a:a + 224, b:b + 224] for a in ys for b in xs])
            crops = torch.cat([crops, torch.flip(crops, dims=[3])])      # 50 crops
            p = model(crops).softmax(dim=1).mean(dim=0, keepdim=True)
        else:
            net = model if protocol == "crop" else dense

            def posterior(t):
                o = net(t)
                # score map averaged in logit space, soft-max after
                return (o.mean(dim=(2, 3)) if o.dim() == 4 else o).softmax(dim=1)

            # original and flipped averaged in posterior space, soft-max before
            p = (posterior(x) + posterior(torch.flip(x, dims=[3]))) / 2
        out.append(p.cpu().numpy())
        if step % max(1, len(loader) // 8) == 0:
            print(f"    {step * batch:>6} / {len(loader.dataset)}   "
                  f"{time.time() - started:5.0f}s", flush=True)

    out = np.concatenate(out)
    if cache:
        np.save(cache, out)            # float32: float16 here would undo the point
    return out


def true_rank(p, labels):
    """Rank of the correct class, 1 = top. Gives top-1 and top-5 without a full
    argsort of a 50000 x 1000 array."""
    return (p > p[np.arange(len(p)), labels][:, None]).sum(1) + 1


def mcnemar(a, b, label, alpha):
    """Paired comparison of two protocols over the same images.

    Only the images the two protocols disagree on carry any information: where
    both were right, or both wrong, neither protocol is favoured. Under the null
    those disagreements split like a fair coin.

    An aggregate difference of 0.1 points can hide 200 genuine disagreements
    that happened to cancel, and two protocols that differ on 5% of images while
    scoring identically are complementary, not equivalent. Only this test
    distinguishes those cases.
    """
    from scipy.stats import binomtest

    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
    verdict = "significant" if p < alpha else ""
    print(f"  {label:<44} agree {int((a == b).sum()):>5}   "
          f"only-1st {only_a:>5}   only-2nd {only_b:>5}   p = {p:.4f}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data", required=True, help="ImageNet val root (1,000 wnid folders)")
    ap.add_argument("--scales", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--protocols", nargs="+", default=["crop", "dense", "multicrop"])
    ap.add_argument("--cache-dir", default=".")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device).eval()
    dense = to_dense(model).to(device).eval()

    # §3.2 is an exact rewriting, not an approximation; confirm before trusting it.
    probe = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        gap = (model(probe) - dense(probe).flatten(1)).abs().max().item()
    assert gap < 1e-4, f"FC-to-conv conversion is not equivalent (gap {gap:.3e})"
    print(f"FC-to-conv conversion verified, max difference {gap:.3e}\n")

    wnids = sorted(os.listdir(args.data))
    assert len(wnids) == 1000, f"expected 1,000 class folders, found {len(wnids)}"
    labels = np.array([y for _, y in ImageNetVal(args.data, None).items])
    print(f"{len(labels)} images, {labels.max() + 1} classes\n")

    P = {}
    for protocol in args.protocols:
        total = None
        for Q in args.scales:
            print(f"{protocol}, Q={Q}")
            cache = os.path.join(args.cache_dir, f"{protocol}_q{Q}.npy")
            p = posteriors(protocol, args.data, model, dense, device, Q, cache=cache)
            total = p.astype(np.float64) if total is None else total + p
        # §4.2 averages the class posteriors across scales; float64 so that
        # summing three float32 arrays does not round into a 0.1-point verdict
        P[protocol] = (total / len(args.scales)).astype(np.float32)

    if "dense" in P and "multicrop" in P:
        P["multicrop & dense"] = (P["multicrop"] + P["dense"]) / 2

    print(f"\nQ = {args.scales}\n")
    ranks = {}
    for name, p in P.items():
        r = ranks[name] = true_rank(p, labels)
        print(f"  {name:<20} top-1 error {100 * (r > 1).mean():6.3f}%   "
              f"top-5 error {100 * (r > 5).mean():6.3f}%")

    pairs = [(a, b) for a, b in [("dense", "multicrop"),
                                 ("dense", "multicrop & dense"),
                                 ("multicrop", "multicrop & dense")]
             if a in ranks and b in ranks]
    if pairs:
        alpha = 0.05 / (2 * len(pairs))
        print(f"\nTable 5, pre-registered: {2 * len(pairs)} tests, "
              f"Bonferroni threshold {alpha:.4f}\n")
        for a, b in pairs:
            for k in (1, 5):
                mcnemar(ranks[a] <= k, ranks[b] <= k, f"top-{k}  {a} vs {b}", alpha)


if __name__ == "__main__":
    main()
