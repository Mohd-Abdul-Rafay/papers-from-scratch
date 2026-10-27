"""
The experiment of Figure 1 and Figure 6: plain nets against residual nets on
CIFAR-10, at two depths.

The claim being tested is the paper's central one, and it is about TRAINING
error, not test error:

    "with the network depth increasing, accuracy gets saturated (which might
     be unsurprising) and then degrades rapidly. Unexpectedly, such
     degradation is not caused by overfitting, and adding more layers to a
     suitably deep model leads to higher TRAINING error"

So a deeper plain net should fit the training set WORSE than a shallower one.
If that does not reproduce, there is nothing for the rest of the paper to fix.

Everything in SETTINGS is quoted from section 4.2. Nothing is tuned.

    python train_cifar.py --models plain20 plain56 resnet20 resnet56
    python train_cifar.py --models plain20 --schedule tail3x
    python train_cifar.py --models plain20 --schedule smoke     # a minute

Paths come from the command line, or from the environment:

    RESNET_DATA     where cifar-10-batches-py lives     (default ./dataset)
    RESNET_OUT      where result JSONs are written      (default ./results)
    RESNET_MIRROR   optional second copy, verified      (default none)

No path is written into this file, so it can be committed as is.

Results go to <out>/<model>_<schedule>.json, or
<out>/<model>_<schedule>_seed<n>.json for any seed other than 0. Each file
records what produced it and on what, so a number can always be traced back to
a run.

One honest departure from the paper: the paper's 45k/5k split was used to pick
the schedule, and that schedule is taken as given here, so training uses all
50,000 images and the reported test error is on the 10,000 test images. This is
recorded in every output file.
"""
import argparse
import copy
import json
import os
import shutil
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cifar_resnet import PAPER_TABLE_6, build

# ---------------------------------------------------------------- settings
# Section 4.2, verbatim:
#   "We use a weight decay of 0.0001 and momentum of 0.9, and adopt the weight
#    initialization in [12] and BN but with no dropout. These models are
#    trained with a mini-batch size of 128 on two GPUs. We start with a
#    learning rate of 0.1, divide it by 10 at 32k and 48k iterations, and
#    terminate training at 64k iterations, which is determined on a 45k/5k
#    train/val split."
SETTINGS = dict(batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4)

SCHEDULES = {
    # name      total iterations, where the learning rate drops
    "paper":   (64000, (32000, 48000)),
    "tail3x":  (192000, (32000, 48000)),    # paper's schedule, then a 144k tail at
                                            # lr 0.001 -- first 64k identical to "paper",
                                            # which is what makes the curves comparable
    "long3x":  (192000, (96000, 144000)),   # the whole schedule stretched 3x
    "short":   (8000, (4000, 6000)),        # same shape, one eighth the length
    "smoke":   (200, (100, 150)),           # just to check the script runs
}


# ---------------------------------------------------------------- data
def loaders(root, batch_size, workers):
    """Section 4.2: 'We follow the simple data augmentation in [24] for
    training: 4 pixels are padded on each side, and a 32x32 crop is randomly
    sampled from the padded image or its horizontal flip. For testing, we only
    evaluate the single view of the original 32x32 image.'

    Three loaders come back, not two. The third is the training set with the
    augmentation switched OFF -- that is what 'training error' has to be
    measured on, because a random crop of a horizontal flip is not the training
    image. It shares the same pixels in memory, so it costs nothing.

    A departure worth stating: section 4.2 subtracts the per-pixel mean. This
    uses the per-channel mean and divides by the per-channel standard
    deviation, which is what every modern reimplementation does. Three numbers
    instead of 3,072.
    """
    train = datasets.CIFAR10(root, train=True, download=True)
    mean = (train.data / 255.0).mean(axis=(0, 1, 2)).tolist()
    std = (train.data / 255.0).std(axis=(0, 1, 2)).tolist()

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train.transform = train_tf
    train_clean = copy.copy(train)          # shallow: shares .data, no second read
    train_clean.transform = eval_tf
    test = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (DataLoader(train, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(train_clean, 256, shuffle=False, **kw),
            DataLoader(test, 256, shuffle=False, **kw),
            mean, std)


@torch.no_grad()
def error_rate(model, loader, device):
    model.eval()
    wrong = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        wrong += (model(x).argmax(1) != y).sum().item()
        total += y.numel()
    return 100.0 * wrong / total


def mirror(path, mirror_dir):
    """Copy a finished result somewhere else and prove the copy arrived.

    This exists because a network-mounted folder can accept a write, report
    success, and never persist it -- which is how one full set of results was
    lost. The byte comparison catches a failed or truncated write. It cannot
    prove the remote service kept the file: for that, look from a fresh session
    or ask the service directly.
    """
    os.makedirs(mirror_dir, exist_ok=True)
    dest = os.path.join(mirror_dir, os.path.basename(path))
    shutil.copyfile(path, dest)
    with open(path, "rb") as a, open(dest, "rb") as b:
        src, cp = a.read(), b.read()
    if cp != src:
        raise RuntimeError(
            f"mirror copy differs from the source: {len(cp):,} bytes at {dest} "
            f"against {len(src):,} bytes at {path}")
    json.loads(cp)                          # must parse, not merely exist
    print(f"  mirrored -> {dest}  ({len(cp):,} bytes, byte-for-byte)")
    return dest


# ---------------------------------------------------------------- training
def train_one(name, schedule, seed, device, train_loader, train_eval_loader,
              test_loader, out_dir, mirror_dir, log_every, norm):
    model, depth = build(name)
    params = sum(p.numel() for p in model.parameters())

    # Both plain and residual have Table 6's count, because option A is
    # parameter-free. Checking it here means a wrong architecture cannot
    # consume an hour of GPU before anyone notices.
    if depth in PAPER_TABLE_6:
        want = PAPER_TABLE_6[depth]
        assert abs(params - want) / want < 0.02, \
            f"{name}: {params:,} parameters, Table 6 says {want:,.0f}"

    model = model.to(device)
    total_iters, drops = SCHEDULES[schedule]
    opt = torch.optim.SGD(model.parameters(), lr=SETTINGS["lr"],
                          momentum=SETTINGS["momentum"],
                          weight_decay=SETTINGS["weight_decay"])
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=list(drops), gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    print(f"\n{'='*70}\n{name}  ({depth} layers, {params:,} parameters)"
          f"  {schedule} schedule: {total_iters:,} iterations  seed {seed}\n{'='*70}")

    history, it, t0 = [], 0, time.time()
    run_wrong = run_seen = 0
    while it < total_iters:
        model.train()
        for x, y in train_loader:
            if it >= total_iters:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            run_wrong += (out.argmax(1) != y).sum().item()
            run_seen += y.numel()
            it += 1

            if it % log_every == 0 or it == total_iters:
                train_err = 100.0 * run_wrong / run_seen
                test_err = error_rate(model, test_loader, device)
                model.train()
                history.append(dict(iteration=it, lr=opt.param_groups[0]["lr"],
                                    train_error=round(train_err, 3),
                                    test_error=round(test_err, 3),
                                    seconds=round(time.time() - t0, 1)))
                print(f"  iter {it:>7,}/{total_iters:,}  lr {opt.param_groups[0]['lr']:<7.4f}"
                      f"  train err {train_err:6.2f}%   test err {test_err:6.2f}%"
                      f"   {time.time()-t0:6.0f}s")
                run_wrong = run_seen = 0

    final_train = error_rate(model, train_eval_loader, device)   # no augmentation
    final_test = error_rate(model, test_loader, device)
    elapsed = time.time() - t0

    record = dict(
        _provenance=dict(
            produced_by="train_cifar.py",
            RECONSTRUCTED=False,
            settings=SETTINGS,
            schedule=dict(name=schedule, iterations=total_iters, lr_drops=list(drops)),
            paper_schedule=(schedule == "paper"),
            seed=seed,
            train_set=("all 50,000 training images; test error on the 10,000 "
                       "test images"),
            normalisation=dict(mean=norm[0], std=norm[1],
                               kind="per-channel mean and std; the paper uses the "
                                    "per-pixel mean"),
            device=(f"cuda - {torch.cuda.get_device_name(0)}"
                    if device.type == "cuda" else "cpu"),
            torch=torch.__version__,
            seconds=round(elapsed, 1),
            note=("train_error inside history is the running error over the training "
                  "batches since the last log -- measured WITH augmentation, in train() "
                  "mode, which is what the paper's curves plot. final_train_error is a "
                  "separate pass over the same 50,000 images with augmentation off, in "
                  "eval() mode, and that is the number the degradation claim is about."),
        ),
        model=name, depth=depth, parameters=params,
        final_train_error=round(final_train, 3),
        final_test_error=round(final_test, 3),
        history=history,
    )

    os.makedirs(out_dir, exist_ok=True)
    stem = f"{name}_{schedule}" if seed == 0 else f"{name}_{schedule}_seed{seed}"
    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=1)
    print(f"  final: train {final_train:.2f}%   test {final_test:.2f}%"
          f"   {elapsed/60:.1f} min   -> {path}")
    if mirror_dir:
        mirror(path, mirror_dir)
    return record


def main(argv=None):
    """argv=None reads sys.argv, which is what you want from a terminal.
    Pass a list when calling this from a notebook cell, where sys.argv carries
    the kernel's own arguments and argparse would refuse them:
        main(["--models", "plain20", "--schedule", "smoke"])
    """
    ap = argparse.ArgumentParser(description="plain vs residual on CIFAR-10")
    ap.add_argument("--models", nargs="+",
                    default=["plain20", "plain56", "resnet20", "resnet56"])
    ap.add_argument("--schedule", choices=sorted(SCHEDULES), default="paper")
    ap.add_argument("--data", default=os.environ.get("RESNET_DATA", "./dataset"))
    ap.add_argument("--out", default=os.environ.get("RESNET_OUT", "./results"))
    ap.add_argument("--mirror", default=os.environ.get("RESNET_MIRROR") or None,
                    help="copy each result here as well, and verify the copy")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="run every model once per seed; seed 0 keeps the plain "
                         "filename, others get a _seed<n> suffix")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_every = args.log_every or max(50, SCHEDULES[args.schedule][0] // 40)

    print(f"device       : {device}"
          f"{' - ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")
    print(f"schedule     : {args.schedule}  {SCHEDULES[args.schedule][0]:,} iterations")
    if args.schedule != "paper":
        print("               NOT the paper's schedule -- recorded as such in the output")
    print(f"seeds        : {args.seeds}")
    print(f"data         : {args.data}")
    print(f"results      : {args.out}")
    print(f"mirror       : {args.mirror or 'none'}")

    train_loader, train_eval_loader, test_loader, mean, std = loaders(
        args.data, SETTINGS["batch_size"], args.workers)
    print(f"normalisation: mean {[round(m, 4) for m in mean]}  "
          f"std {[round(s, 4) for s in std]}")

    done = {}
    for seed in args.seeds:
        for name in args.models:
            torch.manual_seed(seed)
            done[(name, seed)] = train_one(
                name, args.schedule, seed, device, train_loader, train_eval_loader,
                test_loader, args.out, args.mirror, log_every, (mean, std))

    print("\n" + "=" * 70)
    print("THE CLAIM: a deeper PLAIN net should fit the training set WORSE")
    print("=" * 70)
    print(f"  {'model':>10}  {'seed':>4}  {'params':>10}  {'train err':>10}  {'test err':>9}")
    for (name, seed), r in done.items():
        print(f"  {name:>10}  {seed:>4}  {r['parameters']:>10,}"
              f"  {r['final_train_error']:>9.2f}%  {r['final_test_error']:>8.2f}%")

    for deep, shallow, label, expect_positive in (
            ("plain56", "plain20", "plain56 - plain20  ", True),
            ("resnet56", "resnet20", "resnet56 - resnet20", False)):
        pairs = [(done[(deep, s)], done[(shallow, s)]) for s in args.seeds
                 if (deep, s) in done and (shallow, s) in done]
        if not pairs:
            continue
        gaps = [d["final_train_error"] - s["final_train_error"] for d, s in pairs]
        print(f"\n  {label}, TRAINING error, per seed: "
              + "  ".join(f"{g:+.2f}" for g in gaps))
        # One seed buys a direction, never a verdict. Printing "degradation
        # reproduced" under a single run reads exactly like printing it under
        # three, and that is how a one-draw number ends up quoted as a finding.
        if len(gaps) == 1:
            print(f"  {' ' * len(label)}  ONE SEED -- a direction, not a result: "
                  f"there is no spread to measure it against")
            print("  the deeper net fits the training set "
                  + ("worse" if gaps[0] > 0 else "better") + " in this single run")
            continue

        lo, hi = min(gaps), max(gaps)
        print(f"  {' ' * len(label)}  spread {hi - lo:.2f} points over "
              f"{len(gaps)} seeds -- a difference smaller than this is not a result")
        if expect_positive:
            print("  degradation reproduced in every seed" if all(g > 0 for g in gaps)
                  else "  NOT reproduced in every seed")
        else:
            print("  reversed in every seed, as the paper reports"
                  if all(g < 0 for g in gaps) else "  NOT reversed in every seed")
    return done


if __name__ == "__main__":
    main()
