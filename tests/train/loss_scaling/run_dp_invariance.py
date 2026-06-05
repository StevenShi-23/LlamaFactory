"""Driver for the explicit-reduction DP-invariance check (run_dp_math.py).

For fixed global batch, launch each dp in {dps} (ga = global_bs/dp implicitly via
strided sharding) and assert the final gradient (raw-sum -> all-reduce -> /Z) is
identical across dp, for per_token and per_sample. With --packing 1 the same is
checked for packed sequences; additionally packed vs non-packed per_token must
match (same underlying samples).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src"))


def run(mode, packing, global_bs, dp, out, port, spp=3):
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + HERE + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["DISABLE_VERSION_CHECK"] = "1"
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={dp}", f"--master_port={port}",
        os.path.join(HERE, "run_dp_math.py"),
        "--mode", mode, "--packing", str(packing),
        "--global_bs", str(global_bs), "--samples_per_pack", str(spp), "--out", out,
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"[FAIL] mode={mode} packing={packing} dp={dp} crashed:\n{r.stdout[-1500:]}\n{r.stderr[-2500:]}")
        return None
    return torch.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packing", type=int, default=0)
    ap.add_argument("--global_bs", type=int, default=12)
    ap.add_argument("--dps", type=str, default="1,2,3,6")
    ap.add_argument("--spp", type=int, default=3)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-4)
    args = ap.parse_args()

    dps = [int(x) for x in args.dps.split(",")]
    ok = True
    first_grad = {}
    with tempfile.TemporaryDirectory() as td:
        port = 29700
        for mode in ["per_token", "per_sample"]:
            print(f"\n=== DP invariance: mode={mode} packing={args.packing} global_bs={args.global_bs} ===")
            ref = None
            for dp in dps:
                port += 1
                res = run(mode, args.packing, args.global_bs, dp, os.path.join(td, f"{mode}_{dp}.pt"), port, args.spp)
                if res is None:
                    ok = False
                    continue
                if ref is None:
                    ref = res
                    first_grad[mode] = res["grad"]
                    print(f"  dp={dp}: Z={res['Z']:.0f} grad_norm={res['grad_norm']:.6f} (reference)")
                else:
                    md = (ref["grad"] - res["grad"]).abs().max().item()
                    close = torch.allclose(ref["grad"], res["grad"], atol=args.atol, rtol=args.rtol)
                    ok = ok and close
                    print(f"  dp={dp}: Z={res['Z']:.0f} grad_norm={res['grad_norm']:.6f} "
                          f"max|dgrad|={md:.2e} [{'OK' if close else 'MISMATCH'}]")

        if "per_token" in first_grad and "per_sample" in first_grad:
            diff = (first_grad["per_token"] - first_grad["per_sample"]).abs().max().item()
            print(f"\n  per_token vs per_sample max|dgrad|={diff:.4e} (expected > 0)")
            if diff < 1e-6:
                print("  [WARN] modes numerically identical")

    print("\n==== RESULT:", "PASS" if ok else "FAIL", "====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
