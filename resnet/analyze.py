"""
Read the result JSONs and print the comparisons the paper's claims turn on.

This is a script and not a notebook cell on purpose: the notebook cells that
held this analysis were lost with a runtime, and the numbers had to be
transcribed back out of printed output.

    python analyze.py --results ./results

Figures are figures.py's job; this script only reports numbers.

It reports whatever it finds and says plainly what is missing, so a partial set
of runs still gives a partial answer instead of a traceback.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

TABLE_6_TEST = {20: 8.75, 32: 7.51, 44: 7.17, 56: 6.97, 110: 6.43}

NAME = re.compile(r"^(?P<model>plain\d+|resnet\d+)_(?P<sched>\w+?)(?:_seed(?P<seed>\d+))?$")


def load(results_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        m = NAME.match(os.path.splitext(os.path.basename(path))[0])
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        runs[(m["model"], m["sched"], int(m["seed"] or 0))] = r
    return runs


def table(runs):
    print(f"{'model':>9}  {'schedule':>8}  {'seed':>4}  {'params':>9}  "
          f"{'train':>7}  {'test':>7}  {'Table 6':>8}  {'min':>5}  {'src':>6}")
    print(f"{'-'*9}  {'-'*8}  {'-'*4}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*5}  {'-'*6}")
    for (model, sched, seed), r in sorted(runs.items()):
        p = r["_provenance"]
        paper = (f"{TABLE_6_TEST[r['depth']]:.2f}%"
                 if model.startswith("resnet") and sched == "paper"
                 and r["depth"] in TABLE_6_TEST else "-")
        secs = p.get("seconds")
        print(f"{model:>9}  {sched:>8}  {seed:>4}  {r['parameters']:>9,}  "
              f"{r['final_train_error']:>6.2f}%  {r['final_test_error']:>6.2f}%  "
              f"{paper:>8}  {secs/60 if secs else 0:>5.0f}  "
              f"{'REBUILT' if p.get('RECONSTRUCTED') else 'run':>6}")
    if any(r["_provenance"].get("RECONSTRUCTED") for r in runs.values()):
        print("\n  REBUILT = transcribed from printed output, not written by "
              "train_cifar.py.\n  Secondary record: see _provenance.not_recovered "
              "in those files.")


def gaps(runs):
    """The two differences the paper's claims are about, per schedule and seed."""
    out = []
    by_sched = defaultdict(dict)
    for (model, sched, seed), r in runs.items():
        by_sched[sched][(model, seed)] = r

    for sched, group in sorted(by_sched.items()):
        for deep, shallow, label in (("plain56", "plain20", "plain56 - plain20  "),
                                     ("resnet56", "resnet20", "resnet56 - resnet20")):
            seeds = sorted({s for (m, s) in group if m in (deep, shallow)})
            pairs = [(s, group[(deep, s)], group[(shallow, s)]) for s in seeds
                     if (deep, s) in group and (shallow, s) in group]
            if not pairs:
                continue
            print(f"\n  {sched}: {label}")
            for s, d, sh in pairs:
                g = d["final_train_error"] - sh["final_train_error"]
                t = d["final_test_error"] - sh["final_test_error"]
                ratio = (d["final_train_error"] / sh["final_train_error"]
                         if sh["final_train_error"] else float("inf"))
                print(f"    seed {s}:  train {g:+6.2f} points   ({ratio:5.2f}x)"
                      f"   test {t:+6.2f} points")
                out.append((sched, label.strip(), s, g))
            if len(pairs) > 1:
                gs = [d["final_train_error"] - sh["final_train_error"] for _, d, sh in pairs]
                print(f"    spread over {len(gs)} seeds: {max(gs)-min(gs):.2f} points"
                      f"  -- a difference smaller than this is not a result")
            else:
                print("    one seed only: this difference has no error bar yet")
    return out


def budget(runs):
    """What the extra iterations bought, where both schedules exist."""
    pairs = [(m, s) for (m, sch, s) in runs if sch == "paper"
             and (m, "tail3x", s) in runs]
    if not pairs:
        return
    print("\n  what 3x the iterations bought (same lr drops, so the first 64k "
          "is the same experiment)")
    print(f"    {'model':>9}  {'seed':>4}  {'train 64k':>10}  {'train 192k':>11}"
          f"  {'test 64k':>9}  {'test 192k':>10}")
    for m, s in sorted(pairs):
        a, b = runs[(m, "paper", s)], runs[(m, "tail3x", s)]
        print(f"    {m:>9}  {s:>4}  {a['final_train_error']:>9.2f}%"
              f"  {b['final_train_error']:>10.2f}%  {a['final_test_error']:>8.2f}%"
              f"  {b['final_test_error']:>9.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("RESNET_OUT", "./results"))
    args = ap.parse_args()

    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no result JSONs in {args.results}")
    table(runs)
    gaps(runs)
    budget(runs)


if __name__ == "__main__":
    main()
