"""
What Table 2 cannot see — the cost of a VGG layer in time and memory.

Table 2 quantifies cost as parameters, and by that measure some things are free
that are not. Configurations A and A-LRN have identical parameter counts and
identical arithmetic; the paper's only remark on what local response
normalization costs is that it "leads to increased memory consumption and
computation time" (section 4.1), with no figure attached. This script attaches
one.

Three measurements:

  lrn          A against A-LRN on four quantities. Two of them are what Table 2
               reports and are identical. Two of them are not.

  inplace      Whether nn.ReLU(inplace=True) reduces peak memory. The honest
               answer is that it depends entirely on which peak you measure,
               and measuring the wrong one reports exactly zero.

  intensity    The marginal cost of each step up Table 1, in milliseconds per
               GMAC added. A multiply-accumulate is not a multiply-accumulate:
               what it costs depends on the kernel shape it sits in and where
               in the network it sits.

Timing hygiene, because none of this survives without it: warm up first, call
torch.cuda.synchronize() on BOTH sides of the clock, and take the median rather
than the mean. A GPU runs asynchronously — time it without synchronising and you
measure how long it took to queue the work, not to do it.

    python cost.py                 # all three, writes results/cost.json
    python cost.py --only lrn      # one of them
    python cost.py --batch 16      # if 32 does not fit

The committed results/cost.json is a run on one NVIDIA A100-SXM4-80GB with
PyTorch 2.x in float32. Running this script overwrites it with yours. The
absolute milliseconds are hardware; the ratios and the directions are the result.
"""

import argparse
import json
import os
import statistics
import time

import torch
import torch.nn as nn

from vgg import CONFIGS, make_vgg

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def step_time_and_memory(model, batch, size=224, steps=25, warmup=5, device="cuda"):
    """Median forward+backward time in ms, and peak memory in MB.

    Median, not mean: one stray step from a kernel autotune or another process
    moves a mean and does not move a median.
    """
    model = model.to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    x = torch.randn(batch, 3, size, size, device=device)
    y = torch.randint(0, 1000, (batch,), device=device)
    loss_fn = nn.CrossEntropyLoss()

    def one():
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one()
        torch.cuda.synchronize()                 # both sides, or this is queueing time
        times.append((time.perf_counter() - t0) * 1000)

    peak = torch.cuda.max_memory_allocated() / 2**20
    model.cpu()
    torch.cuda.empty_cache()
    return statistics.median(times), peak


def macs(model, size=224):
    """Multiply-accumulates for one image, counted from the real layer shapes."""
    total, x = 0, torch.randn(1, 3, size, size)
    with torch.no_grad():
        for m in model:
            if isinstance(m, nn.Conv2d):
                out = m(x)
                total += (m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                          * m.out_channels * out.shape[2] * out.shape[3])
                x = out
            elif isinstance(m, nn.Linear):
                total += m.in_features * m.out_features
                x = m(x)
            else:
                x = m(x)
    return total


def measure_lrn(batch, device):
    print("\nLRN — what configuration A-LRN costs over A\n")
    out = {}
    for name in ("A", "A-LRN"):
        model = make_vgg(name)
        g = macs(model) / 1e9
        ms, mb = step_time_and_memory(model, batch, device=device)
        params = sum(p.numel() for p in model.parameters())
        out["A" if name == "A" else "A_LRN"] = {
            "params": params, "gmacs": round(g, 2),
            "step_ms": round(ms, 2), "peak_mb": round(mb)}
        print(f"  {name:<6} {params:>12,} params   {g:5.2f} GMACs   "
              f"{ms:6.2f} ms   {mb:7.0f} MB")

    a, l = out["A"], out["A_LRN"]
    out["delta"] = {
        "params": l["params"] - a["params"],
        "gmacs": round(l["gmacs"] - a["gmacs"], 2),
        "step_ms_pct": round(100 * (l["step_ms"] - a["step_ms"]) / a["step_ms"], 1),
        "peak_mb": round(l["peak_mb"] - a["peak_mb"]),
    }
    print(f"\n  parameters   {out['delta']['params']:+,}")
    print(f"  GMACs        {out['delta']['gmacs']:+.2f}")
    print(f"  step time    {out['delta']['step_ms_pct']:+.1f}%")
    print(f"  peak memory  {out['delta']['peak_mb']:+,} MB")
    print("\n  The first two are what Table 2 reports. The last two are what it cannot.")

    # where the memory goes: count the operations PyTorch expands LRN into
    lrn = nn.LocalResponseNorm(size=5, alpha=5e-4, beta=0.75, k=2.0).to(device)
    probe = torch.randn(batch, 64, 224, 224, device=device, requires_grad=True)
    seen, stack = set(), [lrn(probe).grad_fn]
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        stack += [f for f, _ in fn.next_functions]
    tensor_mb = probe.numel() * 4 / 2**20
    out["why"] = {
        "autograd_ops_in_pytorch_lrn": len(seen),
        "one_full_size_tensor_mb": round(tensor_mb),
        "measured_increase_mb": out["delta"]["peak_mb"],
        "ratio": round(out["delta"]["peak_mb"] / tensor_mb, 2),
    }
    print(f"\n  LocalResponseNorm expands into {len(seen)} autograd operations; "
          f"one full-size tensor is {tensor_mb:.0f} MB,")
    print(f"  and the measured increase is {out['why']['ratio']:.2f} of them.")
    return out


def measure_inplace(batch, device):
    print("\nIN-PLACE RELU — which peak you measure decides the answer\n")
    res = {}
    for flag in (False, True):
        model = make_vgg("D")
        for m in model:
            if isinstance(m, nn.ReLU):
                m.inplace = flag
        model = model.to(device).train()
        x = torch.randn(batch, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (batch,), device=device)

        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        loss = nn.CrossEntropyLoss()(model(x), y)
        torch.cuda.synchronize()
        fwd = torch.cuda.max_memory_allocated() / 2**20   # peak so far: forward only
        loss.backward()
        torch.cuda.synchronize()
        glob = torch.cuda.max_memory_allocated() / 2**20  # peak over the whole step
        res[flag] = (fwd, glob)
        model.cpu(); torch.cuda.empty_cache()
        print(f"  inplace={str(flag):<5}  forward peak {fwd:7.1f} MB   global peak {glob:7.1f} MB")

    out = {
        "measured": {"inplace_false_peak_mb": round(res[False][1], 1),
                     "inplace_true_peak_mb": round(res[True][1], 1),
                     "saving_mb": round(res[False][1] - res[True][1], 1)},
        "forward_peak": {"inplace_false_mb": round(res[False][0], 1),
                         "inplace_true_mb": round(res[True][0], 1),
                         "saving_mb": round(res[False][0] - res[True][0], 1)},
        "global_peak_set_by": "backward",
        "global_peak_mb": round(max(res[False][1], res[True][1]), 1),
    }
    print(f"\n  saving at the global peak : {out['measured']['saving_mb']:.1f} MB")
    print(f"  saving at the forward peak: {out['forward_peak']['saving_mb']:.1f} MB")
    print("\n  The saving is real and it is in the forward pass. It is invisible in the")
    print("  global peak because that peak is set during backward, when the stored")
    print("  activations are being consumed anyway. Report only the global peak and you")
    print("  report zero for an effect of several hundred megabytes.")
    return out


def measure_intensity(batch, device):
    print("\nARITHMETIC INTENSITY — the marginal cost of each step up Table 1\n")
    order = ["A", "B", "C", "D", "E"]
    changed = {
        "A to B": "second conv in blocks 1–2 (224², 112²)",
        "B to C": "three 1 × 1 convs (56, 28, 14)",
        "C to D": "the same three, now 3 × 3",
        "D to E": "fourth conv in blocks 3–5",
    }
    point = {}
    for name in order:
        model = make_vgg(name)
        g = macs(model) / 1e9
        ms, _ = step_time_and_memory(model, batch, device=device)
        point[name] = (g, ms)
        print(f"  {name:<3} {g:6.2f} GMACs   {ms:6.2f} ms")

    print(f"\n  {'step':<9}{'+GMACs':>9}{'+ms':>9}{'ms per GMAC':>14}   what changed")
    steps = []
    for a, b in zip(order, order[1:]):
        dg = point[b][0] - point[a][0]
        dm = point[b][1] - point[a][1]
        key = f"{a} to {b}"
        steps.append({"step": key, "gmacs": round(dg, 2), "ms": round(dm, 2),
                      "ms_per_gmac": round(dm / dg, 2), "changed": changed[key]})
        print(f"  {key:<9}{dg:9.2f}{dm:9.2f}{dm/dg:14.2f}   {changed[key]}")

    by = {s["step"]: s["ms_per_gmac"] for s in steps}
    out = {"steps": steps, "two_effects": {
        "kernel_shape": {"one_by_one_ms_per_gmac": by["B to C"],
                         "three_by_three_ms_per_gmac": by["C to D"]},
        "where_in_the_network": {"early_224_112_64_128_channels": by["A to B"],
                                 "late_56_28_14_256_512_channels": by["C to D"]}}}
    print(f"\n  C to D adds the same arithmetic as A to B and costs "
          f"{by['A to B'] / by['C to D']:.1f}x less per GMAC — it is added late, on small")
    print("  maps. And the 1 x 1 layers of configuration C are the worst value in the")
    print("  table: a 3 x 3 kernel does nine multiply-accumulates per weight it loads,")
    print("  a 1 x 1 does one.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only", choices=["lrn", "inplace", "intensity"], nargs="*")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This script measures GPU time and memory and needs CUDA.")
    device = "cuda"
    print(f"{torch.cuda.get_device_name(0)} · PyTorch {torch.__version__} · "
          f"batch {args.batch} · float32")

    want = args.only or ["lrn", "inplace", "intensity"]
    path = os.path.join(RESULTS, "cost.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out["_provenance"] = {
        "produced_by": "vgg/cost.py",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch": args.batch,
        "method": "forward + backward, median of 25 steps after 5 warmup steps, "
                  "synchronised on both sides of the clock; peak from "
                  "torch.cuda.max_memory_allocated() after a reset",
        "note": "Absolute milliseconds are hardware. The ratios and directions are "
                "the result.",
    }
    if "lrn" in want:
        out.setdefault("lrn", {}).update(measure_lrn(args.batch, device))
    if "inplace" in want:
        out.setdefault("inplace_relu", {}).update(measure_inplace(args.batch, device))
    if "intensity" in want:
        out.setdefault("arithmetic_intensity", {}).update(measure_intensity(args.batch, device))

    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
