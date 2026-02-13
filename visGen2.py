#!/usr/bin/env python3
import json, argparse, sys, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPECTED_RATIOS = [(0,10),(1,9),(2,8),(3,7),(4,6),(5,5),(6,4),(7,3),(8,2),(9,1),(10,0)]

def slugify(txt: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', str(txt)).strip('_')

def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)

def collect_from_experiment(exp: dict, source_file: Path, exp_idx: int):
    args = exp.get("arguments", {})
    data = exp.get("data", {})
    dcfg = args.get("dConfig")
    if not dcfg:
        return None, f"{source_file}[exp#{exp_idx}]: missing dConfig"

    test = data.get("Data Shift Test Data", {}) or {}
    runs = test.get("Individual Test Data", []) or []
    if not runs:
        return None, f"{source_file}[exp#{exp_idx}]: no Individual Test Data runs"

    groups = defaultdict(lambda: {"mmds": [], "det": 0, "n": 0})
    for r in runs:
        ss = r.get("Source Samples")
        ts = r.get("Target Samples")
        m = r.get("MMD")
        if ss is None or ts is None or m is None:
            continue
        groups[(ss, ts)]["mmds"].append(m)
        groups[(ss, ts)]["n"] += 1
        if r.get("Shift Detected") is True:
            groups[(ss, ts)]["det"] += 1

    if not groups:
        return None, f"{source_file}[exp#{exp_idx}]: no valid runs with Source/Target/MMD"

    tau = None
    calib = data.get("Calibration", {})
    if isinstance(calib, dict):
        tau = calib.get("Result", {}).get("Tau")
    if tau is None:
        sanity = data.get("Sanity Check", {})
        if isinstance(sanity, dict):
            tau = sanity.get("Results", {}).get("Tau")

    records = []
    for (ss, ts), g in groups.items():
        mmds = g["mmds"]
        if not mmds:
            continue
        avg_mmd = float(np.mean(mmds))
        std_mmd = float(np.std(mmds))
        tpr = (g["det"] / g["n"] * 100.0) if g["n"] else 0.0
        records.append({
            "dconfig": dcfg,
            "src_samples": ss,
            "tgt_samples": ts,
            "avg_mmd": avg_mmd,
            "std_mmd": std_mmd,
            "tpr": tpr,
            "tau": tau,
            "runs": g["n"],
            "source_file": str(source_file),
            "exp_idx": exp_idx,
        })
    if not records:
        return None, f"{source_file}[exp#{exp_idx}]: no records after grouping"
    return records, None

def plot_dconfig(dcfg, items, outdir: Path, warn_missing: list):
    agg = {}
    for it in items:
        key = (it["src_samples"], it["tgt_samples"])
        n = it["runs"]
        mean = it["avg_mmd"]
        std = it["std_mmd"]
        s = mean * n
        ssq = (std**2 + mean**2) * n
        if key not in agg:
            agg[key] = {"sum":0.0,"sumsq":0.0,"n":0,"dets":0.0,"taus":[]}
        agg[key]["sum"] += s
        agg[key]["sumsq"] += ssq
        agg[key]["n"] += n
        agg[key]["dets"] += (it["tpr"]/100.0) * n
        if it["tau"] is not None:
            agg[key]["taus"].append(it["tau"])

    rows = []
    for (ss, ts), g in agg.items():
        n = g["n"]
        mean = g["sum"] / n
        var = max(g["sumsq"]/n - mean**2, 0.0)
        std = var**0.5
        tpr = (g["dets"] / n) * 100.0
        tau = float(np.mean(g["taus"])) if g["taus"] else None
        rows.append({"src": ss, "tgt": ts, "avg_mmd": mean, "std_mmd": std, "tpr": tpr, "tau": tau, "n": n})

    missing = [pair for pair in EXPECTED_RATIOS if pair not in agg]
    if missing:
        warn_missing.append(f"{dcfg}: missing ratios -> " + ", ".join([f"{s}:{t}" for s,t in missing]))

    rows.sort(key=lambda r: EXPECTED_RATIOS.index((r["src"], r["tgt"])) if (r["src"], r["tgt"]) in EXPECTED_RATIOS else 999)
    if not rows:
        return

    x = np.arange(len(rows))
    labels = [f"{r['src']}:{r['tgt']}" for r in rows]
    tpr = [r["tpr"] for r in rows]
    avg = [r["avg_mmd"] for r in rows]
    std = [r["std_mmd"] for r in rows]
    tau = [r["tau"] if r["tau"] is not None else None for r in rows]

    fig = plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1,2,1)
    bars = ax1.bar(x, tpr, color='#D62828', edgecolor='black')
    for b, v in zip(bars, tpr):
        ax1.text(b.get_x()+b.get_width()/2., b.get_height()+1, f'{v:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_title('TPR (%)'); ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha='right'); ax1.set_ylim(0,110); ax1.axhline(100, ls='--', c='green', alpha=0.5)

    ax2 = plt.subplot(1,2,2)
    ax2.errorbar(x, avg, yerr=std, fmt='o-', color='#F18F01', label='Avg MMD ± std')
    if any(t is not None for t in tau):
        tau_vals = [t if t is not None else 0 for t in tau]
        ax2.plot(x, tau_vals, 's--', color='#2E86AB', alpha=0.7, label='Tau (where present)')
    ax2.set_title('MMD'); ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=45, ha='right'); ax2.legend()

    plt.suptitle(f"{dcfg} | Experiments={len(rows)}", fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93])

    base = f"{slugify(dcfg)}__n{len(rows)}"
    for fmt, dpi in [('png',300), ('svg',None), ('pdf',None), ('jpeg',300)]:
        subdir = outdir / fmt
        subdir.mkdir(parents=True, exist_ok=True)
        out = subdir / f"{base}.{fmt}"
        plt.savefig(out, dpi=dpi if dpi else None, bbox_inches='tight', format=fmt)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-path', nargs='+', required=True)
    ap.add_argument('--recursive', action='store_true')
    ap.add_argument('--pattern', default='*.json')
    ap.add_argument('--output-dir', default='figures_json_per_dconfig')
    ap.add_argument('--expect-dconfigs', type=int, default=None, help='Abort if dConfig count differs')
    args = ap.parse_args()

    json_files = []
    for root in args.log_path:
        p = Path(root)
        if not p.exists(): continue
        json_files.extend(list(p.rglob(args.pattern)) if args.recursive else [f for f in p.iterdir() if f.match(args.pattern)])

    errors = []
    by_dcfg = defaultdict(list)

    for jf in json_files:
        try:
            obj = load_json(jf)
        except Exception as e:
            errors.append(f"{jf}: failed to parse JSON ({e})")
            continue
        exps = obj.get("experiments", [])
        if not exps:
            continue
        for i, exp in enumerate(exps):
            recs, err = collect_from_experiment(exp, jf, i)
            if err:
                if "missing dConfig" not in err:
                    errors.append(err)
                continue
            by_dcfg[recs[0]["dconfig"]].extend(recs)

    if errors:
        print(f"\n⚠️ Issues found ({len(errors)}):")
        for e in errors[:50]:
            print("  " + e)
        if len(errors) > 50:
            print(f"  ... and {len(errors)-50} more")

    dcfg_list = sorted(by_dcfg.keys())
    if args.expect_dconfigs is not None and len(dcfg_list) != args.expect_dconfigs:
        print(f"\n❌ Abort: found {len(dcfg_list)} dConfigs, expected {args.expect_dconfigs}. dConfigs: {dcfg_list}")
        sys.exit(1)

    outdir = Path(args.output_dir)
    warn_missing = []
    for dcfg, items in by_dcfg.items():
        try:
            plot_dconfig(dcfg, items, outdir, warn_missing)
        except Exception as e:
            print(f"\n❌ Abort for {dcfg}: {e}")
            sys.exit(1)

    if warn_missing:
        print("\n⚠️ Missing ratios detected:")
        for w in warn_missing:
            print("  " + w)

    print(f"\n✓ Done. Figures in {outdir}/[png|svg|pdf|jpeg]")

if __name__ == "__main__":
    main()