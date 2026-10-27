#!/usr/bin/env python3
"""
Rebuild train_cifar.py's result JSONs from a notebook cell's printed output.

This exists because a Colab runtime disconnected before its results were copied
off the local disk, while the notebook's own output survived. The numbers here
are the real measured numbers, but the FILE is a transcription, not a primary
record -- so every file it writes says so in _provenance, and names what the
transcription could not recover.

    python rebuild_from_log.py log.txt --out results/

Parses lines of the shape produced by train_cifar.py:

    plain20  (20 layers, 269,722 parameters)  paper schedule: 64,000 iterations
      iter   1,600/64,000  lr 0.1000   train err  46.39%   test err  34.53%       22s
      final: train 0.58%   test 9.44%   13.8 min   -> .../plain20_paper.json
"""
import argparse
import json
import os
import re

# Must match train_cifar.py. Repeated rather than imported so this script can
# run anywhere the log lands.
SETTINGS = dict(batch_size=128, lr=0.1, momentum=0.9, weight_decay=1e-4)
SCHEDULES = {
    "paper":  (64000, (32000, 48000)),
    "tail3x": (192000, (32000, 48000)),
    "long3x": (192000, (96000, 144000)),
    "short":  (8000, (4000, 6000)),
    "smoke":  (200, (100, 150)),
}

RE_DEVICE = re.compile(r"^device\s*:\s*(.+?)\s*$")
RE_NORM = re.compile(r"^normalisation:\s*mean \[(.+?)\]\s+std \[(.+?)\]")
RE_HEAD = re.compile(
    r"^(?P<model>\w+)\s+\((?P<depth>\d+) layers, (?P<params>[\d,]+) parameters\)"
    r"\s+(?P<sched>\w+) schedule: (?P<iters>[\d,]+) iterations\s*$")
RE_ITER = re.compile(
    r"^\s*iter\s+(?P<it>[\d,]+)/(?P<tot>[\d,]+)\s+lr (?P<lr>[\d.]+)\s+"
    r"train err\s+(?P<tr>[\d.]+)%\s+test err\s+(?P<te>[\d.]+)%\s+(?P<secs>[\d.]+)s\s*$")
RE_FINAL = re.compile(
    r"^\s*final:\s*train (?P<tr>[\d.]+)%\s+test (?P<te>[\d.]+)%"
    r"(?:\s+(?P<mins>[\d.]+) min)?\s*->\s*(?P<path>\S+)\s*$")

LOST = [
    "torch version -- not printed by the run",
    "the seed -- not printed; train_cifar.py's default is 0",
    "full float precision of train_error/test_error: the log prints 2 decimals, "
    "the original JSON stored 3",
    "the exact learning-rate floats: the log prints 0.0010, the optimizer held "
    "0.0010000000000000002",
    "the clean-pass train/test errors are kept, but the intermediate per-batch "
    "state that produced them is not recoverable",
]


def parse(text):
    device = None
    mean = std = None
    records, cur = [], None

    for line in text.splitlines():
        m = RE_DEVICE.match(line)
        if m:
            device = m.group(1)
            continue

        m = RE_NORM.match(line)
        if m:
            mean = [float(v) for v in m.group(1).split(",")]
            std = [float(v) for v in m.group(2).split(",")]
            continue

        m = RE_HEAD.match(line)
        if m:
            cur = dict(model=m.group("model"),
                       depth=int(m.group("depth")),
                       parameters=int(m.group("params").replace(",", "")),
                       schedule=m.group("sched"),
                       iterations=int(m.group("iters").replace(",", "")),
                       history=[])
            records.append(cur)
            continue

        m = RE_ITER.match(line)
        if m and cur is not None:
            cur["history"].append(dict(
                iteration=int(m.group("it").replace(",", "")),
                lr=float(m.group("lr")),
                train_error=float(m.group("tr")),
                test_error=float(m.group("te")),
                seconds=float(m.group("secs")),      # extra, not in the original schema
            ))
            continue

        m = RE_FINAL.match(line)
        if m and cur is not None:
            cur["final_train_error"] = float(m.group("tr"))
            cur["final_test_error"] = float(m.group("te"))
            cur["minutes"] = float(m.group("mins")) if m.group("mins") else None
            cur["original_path"] = m.group("path")
            cur = None

    return records, device, mean, std


def to_json(r, device, mean, std, source):
    sched = r["schedule"]
    total, drops = SCHEDULES[sched]
    if total != r["iterations"]:
        raise AssertionError(
            f"{r['model']}: log says {r['iterations']:,} iterations, "
            f"SCHEDULES['{sched}'] says {total:,}")
    if not r["history"]:
        raise AssertionError(f"{r['model']}: no iteration lines found")
    if "final_train_error" not in r:
        raise AssertionError(
            f"{r['model']}: no 'final:' line -- the run did not finish, "
            f"or the output was truncated before it")
    if r["history"][-1]["iteration"] != total:
        raise AssertionError(
            f"{r['model']}: last logged iteration is "
            f"{r['history'][-1]['iteration']:,}, expected {total:,}")

    return dict(
        _provenance=dict(
            produced_by="rebuild_from_log.py",
            RECONSTRUCTED=True,
            reconstructed_from=source,
            why=("the Colab runtime disconnected before these results were copied "
                 "off its local disk; the notebook's printed output survived and "
                 "was transcribed back. The numbers are the measured numbers. "
                 "This FILE was not written by train_cifar.py and is a secondary "
                 "record -- prefer a re-run if anything depends on it."),
            not_recovered=LOST,
            settings=SETTINGS,
            schedule=dict(name=sched, iterations=total, lr_drops=list(drops)),
            paper_schedule=(sched == "paper"),
            train_set=("all 50,000 training images; test error on the 10,000 "
                       "test images"),
            device=device,
            normalisation=dict(mean=mean, std=std),
            torch=None,
            seconds=round(r["minutes"] * 60, 1) if r["minutes"] else None,
            note=("train_error inside history is the running error over the training "
                  "batches since the last log -- measured WITH augmentation, in train() "
                  "mode, which is what the paper's curves plot. final_train_error is a "
                  "separate pass over the same 50,000 images with augmentation off, in "
                  "eval() mode, and that is the number the degradation claim is about. "
                  "'seconds' inside each history entry is wall clock since the run "
                  "started; train_cifar.py does not record it, the printed log did."),
        ),
        model=r["model"], depth=r["depth"], parameters=r["parameters"],
        final_train_error=r["final_train_error"],
        final_test_error=r["final_test_error"],
        history=[{k: v for k, v in h.items()} for h in r["history"]],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    text = open(args.log).read()
    records, device, mean, std = parse(text)
    if not records:
        raise SystemExit("no model headers found in the log -- nothing to rebuild")

    os.makedirs(args.out, exist_ok=True)
    source = f"notebook cell output, transcribed from {os.path.basename(args.log)}"

    for r in records:
        rec = to_json(r, device, mean, std, source)
        name = f"{r['model']}_{r['schedule']}.json"
        path = os.path.join(args.out, name)
        with open(path, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"{name:<26} {len(rec['history']):>3} history points  "
              f"train {rec['final_train_error']:>6.2f}%  "
              f"test {rec['final_test_error']:>6.2f}%  -> {path}")

    print(f"\ndevice        : {device}")
    print(f"normalisation : mean {mean}  std {std}")


if __name__ == "__main__":
    main()
