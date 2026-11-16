"""Run inspection (and optionally a cheap fingerprint) over a list of models.

    python tools/corpus.py tools/corpus.txt --fingerprint screen --out corpus.json

Prints a Markdown table.  ``--fingerprint search`` runs the exhaustive seed
search per model (minutes to hours each on CPU).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from gpu_forensics.engine import fingerprint
from gpu_forensics.inspector import inspect
from gpu_forensics.report import dumps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("listing", help="file with one model per line (# comments allowed)")
    parser.add_argument("--fingerprint", choices=["none", "screen", "search"], default="none")
    parser.add_argument("--search-budget", type=float, default=3600)
    parser.add_argument("--out")
    args = parser.parse_args()
    models = [line.split("#")[0].strip() for line in open(args.listing)]
    results = []
    print("| model | verdict | embedding | padding rows | init-like rows | result |")
    print("|---|---|---|---|---|---|")
    for m in filter(None, models):
        try:
            insp = inspect(m)
        except Exception as exc:  # report and continue
            print(f"| {m} | error | | | | {exc} |")
            continue
        row = {"model": m, "inspection": asdict(insp)}
        e = insp.embedding or {}
        result = ""
        rows = ""
        if args.fingerprint != "none" and insp.feasibility["verdict"] != "infeasible":
            rep = fingerprint(m, seed=args.fingerprint, search_budget=args.search_budget, quiet=True,
                              null_seeds=50)
            row["report"] = json.loads(dumps(rep))
            sel = (rep.sample.selection if rep.sample else {}) or {}
            rows = f"{len(rep.sample.rows) if rep.sample else 0} (std {sel.get('init_scale') or 0:.4g})"
            result = rep.classification + (f": {rep.best.devices}, seed {rep.best.seed}" if rep.best and
                                           rep.classification != "inconclusive" else f" ({rep.reason[:60]})")
        verdict = insp.feasibility["verdict"]
        blockers = "; ".join(insp.feasibility.get("blockers", []))
        print(f"| {m} | {verdict}{' — ' + blockers if blockers else ''} | {e.get('dtype', '')} "
              f"{e.get('shape', '')} | {e.get('padding_rows', '')} | {rows} | {result} |", flush=True)
        results.append(row)
    if args.out:
        with open(args.out, "w") as f:
            f.write(dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
