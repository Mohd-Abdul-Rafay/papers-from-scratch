"""
Check the result JSONs against themselves, and reprint the summary that
train_cifar.py printed into the notebook.

Two jobs:

1.  Per file, check what the file claims against what it contains -- the
    filename against the recorded model/schedule/seed, the depth against the
    parameter count, the history against the schedule, the learning rate
    against the drops. A JSON that disagrees with itself is the kind of thing
    nobody notices until a number from it is already in a write-up.

2.  Recompute the "THE CLAIM" block exactly as train_cifar.py prints it, so it
    can be put next to the notebook output and read line for line. The summary
    is derived from the same values the JSON stores, so the two must agree; if
    they do not, one of them was edited.

    python verify.py --results ./results
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

SCHEDULES = {
    "paper":  (64000, (32000, 48000)),
    "tail3x": (192000, (32000, 48000)),
    "long3x": (192000, (96000, 144000)),
    "short":  (8000, (4000, 6000)),
    "smoke":  (200, (100, 150)),
}

NAME = re.compile(r"^(?P<model>plain\d+|resnet\d+)_(?P<sched>\w+?)(?:_seed(?P<seed>\d+))?$")


def expected_parameters(depth):
    """The §4.2 architecture's count, from the shapes alone -- no torch needed.

    first conv 3->16 (3*3*3*16) + BN(16)
    stage s with n blocks, cin -> cout: first block cin->cout then cout->cout,
    remaining blocks cout->cout twice; every conv followed by BN.
    final fc 64 -> 10 with bias.
    """
    n = (depth - 2) // 6
    total = 3 * 3 * 3 * 16 + 2 * 16
    cin = 16
    for cout in (16, 32, 64):
        for b in range(n):
            a = cin if b == 0 else cout
            total += 3 * 3 * a * cout + 2 * cout        # conv1 + BN
            total += 3 * 3 * cout * cout + 2 * cout     # conv2 + BN
        cin = cout
    return total + 64 * 10 + 10


def check_file(path, problems):
    name = os.path.splitext(os.path.basename(path))[0]
    m = NAME.match(name)
    if not m:
        return None
    with open(path) as f:
        r = json.load(f)
    p = r["_provenance"]
    model, sched, seed = m["model"], m["sched"], int(m["seed"] or 0)

    def bad(msg):
        problems.append(f"{name}: {msg}")

    if r["model"] != model:
        bad(f"filename says {model}, file says {r['model']}")
    if p["schedule"]["name"] != sched:
        bad(f"filename says {sched}, file says {p['schedule']['name']}")
    if "seed" in p and p["seed"] != seed:
        bad(f"filename says seed {seed}, file says seed {p['seed']}")

    depth = int(re.sub(r"\D", "", model))
    if r["depth"] != depth:
        bad(f"model name gives depth {depth}, file says {r['depth']}")
    want = expected_parameters(depth)
    if r["parameters"] != want:
        bad(f"depth {depth} should hold {want:,} parameters, file says "
            f"{r['parameters']:,}")

    if sched in SCHEDULES:
        iters, drops = SCHEDULES[sched]
        if p["schedule"]["iterations"] != iters:
            bad(f"schedule {sched} is {iters:,} iterations, file says "
                f"{p['schedule']['iterations']:,}")
        if list(p["schedule"]["lr_drops"]) != list(drops):
            bad(f"schedule {sched} drops at {list(drops)}, file says "
                f"{p['schedule']['lr_drops']}")
        if p.get("paper_schedule") != (sched == "paper"):
            bad(f"paper_schedule is {p.get('paper_schedule')} for schedule {sched}")

    h = r["history"]
    if not h:
        bad("empty history")
        return r
    if h[-1]["iteration"] != p["schedule"]["iterations"]:
        bad(f"history ends at iteration {h[-1]['iteration']:,}, schedule says "
            f"{p['schedule']['iterations']:,}")

    # the learning rate recorded at each point must match the drops
    d = p["schedule"]["lr_drops"]
    for q in h:
        want_lr = 0.1 if q["iteration"] < d[0] else (0.01 if q["iteration"] < d[1] else 0.001)
        if abs(q["lr"] - want_lr) > 1e-9:
            bad(f"iteration {q['iteration']:,} records lr {q['lr']} but the drops "
                f"{d} give {want_lr}")
            break

    for field in ("final_train_error", "final_test_error"):
        if not 0.0 <= r[field] <= 100.0:
            bad(f"{field} is {r[field]}, outside 0-100")
    if not p.get("seconds"):
        bad("no wall clock recorded")
    return r


def summary(runs, schedule):
    """train_cifar.py's own closing block, recomputed from the files."""
    seeds = sorted({s for (_, c, s) in runs if c == schedule})
    print("\n" + "=" * 70)
    print(f"THE CLAIM, recomputed from the JSONs  —  schedule: {schedule}")
    print("=" * 70)
    print(f"  {'model':>10}  {'seed':>4}  {'params':>10}  {'train err':>10}  {'test err':>9}")
    for (model, c, seed), r in sorted(runs.items()):
        if c != schedule:
            continue
        print(f"  {model:>10}  {seed:>4}  {r['parameters']:>10,}"
              f"  {r['final_train_error']:>9.2f}%  {r['final_test_error']:>8.2f}%")

    for deep, shallow, label, expect_positive in (
            ("plain56", "plain20", "plain56 - plain20  ", True),
            ("resnet56", "resnet20", "resnet56 - resnet20", False)):
        pairs = [(runs[(deep, schedule, s)], runs[(shallow, schedule, s)]) for s in seeds
                 if (deep, schedule, s) in runs and (shallow, schedule, s) in runs]
        if not pairs:
            continue
        gaps = [d["final_train_error"] - s["final_train_error"] for d, s in pairs]
        print(f"\n  {label}, TRAINING error, per seed: "
              + "  ".join(f"{g:+.2f}" for g in gaps))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESNET_MIRROR") or "./results")
    args = ap.parse_args()

    problems, runs = [], {}
    for path in sorted(glob.glob(os.path.join(args.results, "*.json"))):
        name = os.path.splitext(os.path.basename(path))[0]
        m = NAME.match(name)
        if not m:
            continue
        r = check_file(path, problems)
        if r:
            runs[(m["model"], m["sched"], int(m["seed"] or 0))] = r

    if not runs:
        raise SystemExit(f"no result JSONs in {args.results}")

    print(f"{len(runs)} files checked in {args.results}")
    rebuilt = sum(1 for r in runs.values() if r["_provenance"].get("RECONSTRUCTED"))
    print(f"  {len(runs) - rebuilt} written by train_cifar.py, {rebuilt} rebuilt "
          f"from notebook output")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for q in problems:
            print(f"  {q}")
    else:
        print("  every file agrees with its own filename, depth, schedule and "
              "learning-rate drops")

    for schedule in sorted({c for (_, c, _) in runs},
                           key=lambda s: list(SCHEDULES).index(s)
                           if s in SCHEDULES else 99):
        summary(runs, schedule)

    print("\nPut each block above next to the matching cell output in the notebook.")
    print("They are computed from the same values the JSON stores, so any")
    print("difference beyond rounding means one of the two was edited.")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
