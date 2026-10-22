"""
Deciding whether a difference between two protocols is real.

dense_eval.py produces a posterior matrix per protocol. This script does the
statistics on them, and it exists because of a specific trap.

Dense evaluation and centre-cropping score 14.73% and 14.88% top-1 on the same
3,925 images. A difference of 0.15 points, six images — easy to explain, and the
explanation would be fiction. The two protocols actually disagree on 196 images:
dense gets 95 right that cropping misses, cropping gets 101 right that dense
misses, and the two cancel. The aggregate number hides both facts — that neither
is better, and that they are very different predictors. Which is exactly why the
paper can say they are COMPLEMENTARY and that combining them beats each alone.

So: a paired test, over per-image outcomes.

  mcnemar     Of the images two protocols disagree on, how lopsided is the
              split? Images where both are right, or both wrong, carry no
              information about which is better and are discarded. Under the
              null the rest split like a fair coin.

  ranks       Where in the ranking a protocol acts. Top-1 scores rank 2 and rank
              900 identically, so a change that moves answers up the list without
              moving them to the front is invisible to it.

  power       How many images a claim of a given size needs. Run before
              concluding that a non-significant result means no effect — on
              3,925 images none of these comparisons reach significance and on
              50,000 almost all of them do, with the same code and the same
              weights.

Comparisons are declared on the command line BEFORE the numbers are looked at,
and Bonferroni-corrected for how many were declared. Running every pair and then
reporting the ones under 0.05 is a different and much weaker thing.

    python analysis.py --posteriors results/posteriors --labels results/labels.npy
    python analysis.py ... --compare dense:multicrop dense:combination
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def true_rank(p, labels):
    """Rank of the correct class, 1 = top, without argsorting 1,000 columns."""
    return (p > p[np.arange(len(p)), labels][:, None]).sum(1) + 1


def mcnemar(a, b, label, alpha):
    """a and b are boolean arrays: was this image correct under each protocol?"""
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = binomtest(only_b, n, 0.5).pvalue if n else 1.0
    sig = p < alpha
    print(f"  {label:<46} agree {int((a == b).sum()):>5}   "
          f"only-1st {only_a:>5}   only-2nd {only_b:>5}   "
          f"p = {p:.4f}{'   significant' if sig else ''}")
    return {"comparison": label, "only_a": only_a, "only_b": only_b,
            "agree": int((a == b).sum()), "p": round(float(p), 6),
            "significant": bool(sig)}   # numpy bool is not JSON serialisable


def rank_migration(r_a, r_b, name_a, name_b):
    """Three-by-three: where the correct class sat before, and where it sits now."""
    def bucket(r):
        return np.where(r == 1, 0, np.where(r <= 5, 1, 2))
    ba, bb = bucket(r_a), bucket(r_b)
    m = [[int(((ba == i) & (bb == j)).sum()) for j in range(3)] for i in range(3)]
    names = ["rank 1", "rank 2-5", "rank 6+"]

    print(f"\n  rows {name_a}, columns {name_b}\n")
    print("  " + " " * 10 + "".join(f"{n:>10}" for n in names))
    for i, row in enumerate(m):
        print(f"  {names[i]:<10}" + "".join(f"{v:>10,}" for v in row))

    entered = int(((ba == 2) & (bb <= 1)).sum())
    to_top = int(((ba == 2) & (bb == 0)).sum())
    left = int(((ba <= 1) & (bb == 2)).sum())
    fell_out = m[0][2]
    print(f"\n  entered the top five: {entered}  "
          f"({to_top} to rank 1, {entered - to_top} to ranks 2-5)")
    print(f"  left the top five:    {left}")
    print(f"  ranked first by {name_a} and below five under {name_b}: {fell_out}")

    d = np.log(r_b) - np.log(r_a)
    moved = d != 0
    w = wilcoxon(d[moved]) if moved.any() else None
    print(f"\n  mean log-rank  {name_a} {np.log(r_a).mean():.4f}   "
          f"{name_b} {np.log(r_b).mean():.4f}")
    if w is not None:
        print(f"  Wilcoxon over the {int(moved.sum()):,} images whose rank changed: "
              f"p = {w.pvalue:.4f}   improved {int((d < 0).sum())}  "
              f"worsened {int((d > 0).sum())}")
        print("  (mean log-rank is pulled by magnitude; the signed test is the honest one)")
    return {"matrix_rows_a_cols_b": m, "entered_top5": entered,
            "entered_landing_at_rank_1": to_top, "left_top5": left,
            "ranked_first_then_below_five": fell_out,
            "mean_log_rank": {name_a: float(np.log(r_a).mean()),
                              name_b: float(np.log(r_b).mean()),
                              "wilcoxon_p": float(w.pvalue) if w is not None else None,
                              "improved": int((d < 0).sum()),
                              "worsened": int((d > 0).sum())}}


def power(discordant, effect_points, total_images, alpha=0.05, want=0.80):
    """How many discordant pairs a claim of this size needs, and how many images.

    The discordant rate is a property of the two protocols, so it is measured
    here rather than assumed, and the required count follows from the split the
    stated effect implies.
    """
    z = {0.05: 1.959964, 0.10: 1.644854}[alpha]
    zb = {0.80: 0.841621, 0.90: 1.281552}[want]
    rate = discordant / total_images
    net = effect_points / 100
    pi = (rate + net) / 2 / rate                      # share of discordant pairs won
    need = (z + zb) ** 2 / (4 * (pi - 0.5) ** 2)
    images = need / rate
    print(f"\n  discordant pairs available : {discordant:,} "
          f"({100 * rate:.2f}% of {total_images:,} images)")
    print(f"  to detect {effect_points:.2f} points at alpha {alpha}, power {want}:")
    print(f"    discordant pairs needed  : {need:.0f}")
    print(f"    images needed            : {images:,.0f}  "
          f"({images / total_images:.1f}x this sample)")
    return {"discordant_pairs_available": discordant,
            "discordant_pairs_needed": round(need),
            "images_needed": round(images),
            "effect_points": effect_points, "alpha": alpha, "power": want}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--posteriors", default=os.path.join(RESULTS, "posteriors"),
                    help="directory of <protocol>.npy posterior matrices from dense_eval.py")
    ap.add_argument("--labels", default=os.path.join(RESULTS, "labels.npy"))
    ap.add_argument("--compare", nargs="*", default=None,
                    help="pairs as a:b, declared before the numbers are seen. "
                         "Default: every pair, which is exploratory and labelled so.")
    ap.add_argument("--ranks", default=None, metavar="A:B",
                    help="rank migration for one pair")
    ap.add_argument("--power-effect", type=float, default=0.4,
                    help="effect size in points to compute required sample for")
    args = ap.parse_args()

    labels = np.load(args.labels)
    P = {os.path.splitext(os.path.basename(f))[0]: np.load(f)
         for f in sorted(glob.glob(os.path.join(args.posteriors, "*.npy")))}
    if not P:
        raise SystemExit(f"no posterior matrices in {args.posteriors} — run dense_eval.py first")
    if "multicrop" in P and "dense" in P and "combination" not in P:
        P["combination"] = (P["multicrop"] + P["dense"]) / 2

    R = {k: true_rank(v, labels) for k, v in P.items()}
    print(f"{len(labels):,} images · protocols: {', '.join(sorted(P))}\n")
    for k in sorted(R):
        r = R[k]
        print(f"  {k:<20} top-1 error {100 * (r > 1).mean():6.3f}%   "
              f"top-5 error {100 * (r > 5).mean():6.3f}%")

    pairs = ([tuple(c.split(":")) for c in args.compare] if args.compare
             else list(itertools.combinations(sorted(P), 2)))
    declared = args.compare is not None
    alpha = 0.05 / (2 * len(pairs))
    print(f"\n{'PRE-REGISTERED' if declared else 'EXPLORATORY — every pair, not declared in advance'}")
    print(f"{2 * len(pairs)} tests, Bonferroni threshold {alpha:.4f}\n")

    out = {"images": int(len(labels)), "pre_registered": declared,
           "tests": 2 * len(pairs), "bonferroni_alpha": round(alpha, 6), "results": []}
    for a, b in pairs:
        for k in (1, 5):
            out["results"].append(mcnemar(R[a] <= k, R[b] <= k, f"top-{k}  {a} vs {b}", alpha))

    if args.ranks:
        a, b = args.ranks.split(":")
        print(f"\nRANK MIGRATION")
        out["rank_migration"] = rank_migration(R[a], R[b], a, b)
        disc = out["rank_migration"]["matrix_rows_a_cols_b"]
        d1 = int(((R[a] <= 1) != (R[b] <= 1)).sum())
        print("\nPOWER")
        out["power"] = power(d1, args.power_effect, len(labels))

    path = os.path.join(RESULTS, "analysis.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
