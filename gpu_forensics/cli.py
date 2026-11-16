"""``gpu-forensics`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, hub
from .report import dumps, render


def _emit(data: dict, output: str | None, fmt: str):
    text = dumps(data)
    if output:
        Path(output).write_text(text + "\n")
    print(render(json.loads(text)) if fmt == "text" else text)


def _ints(value: str) -> tuple[int, ...]:
    out = []
    for part in value.split(","):
        if ":" in part:  # start:stop:step
            a, b, *c = (int(x) for x in part.split(":"))
            out += list(range(a, b, c[0] if c else 4))
        elif part:
            out.append(int(part))
    return tuple(out)


def cmd_inspect(args):
    from .inspector import inspect

    result = inspect(args.model, args.revision, deep=args.deep, tokenizer=not args.no_tokenizer)
    print(dumps(result))
    if args.output:
        Path(args.output).write_text(dumps(result) + "\n")


def cmd_fingerprint(args):
    from .engine import fingerprint

    report = fingerprint(args.model, args.revision, seed=args.seed, dist=args.dist, tensor_name=args.tensor,
                         rows_spec=args.rows, offsets=_ints(args.offsets), tps=_ints(args.tp),
                         search_budget=args.search_budget, backend=args.backend, null_seeds=args.null_seeds,
                         max_rows=args.max_rows, quiet=args.quiet)
    _emit(report, args.output, args.format)
    return 0 if report.classification != "inconclusive" else 3


def cmd_validate(args):
    from .engine import revalidate

    report = json.loads(Path(args.report).read_text())
    result = revalidate(report, null_seeds=args.null_seeds, fresh_rows=args.fresh_rows, quiet=args.quiet)
    _emit(result, args.output, args.format)
    return 0 if result.get("classification") != "inconclusive" else 3


def cmd_search(args):
    from . import sampler, search
    from .engine import parse_rows, search_rows
    from .layouts import catalogue_threads

    ref, _ = hub.resolve(hub.parse(args.model, args.revision))
    from .inspector import inspect

    insp = inspect(args.model, args.revision, tokenizer=False)
    name = args.tensor or (insp.embedding or {}).get("name")
    e = insp.embedding
    if not e or e["name"] != name:
        raise SystemExit("pass --tensor matching the embedding (others need an explicit inventory)")
    tensor = hub.TensorInfo(e["name"], e["dtype"], tuple(e["shape"]), e["file"], *e["byte_range"])
    rows = search_rows(parse_rows(args.rows, tensor.shape[0]), tensor.shape)
    observed, _ = sampler.fetch_rows(ref, tensor, rows)
    threads = _ints(args.threads) if args.threads else catalogue_threads()
    hyps = search.hypotheses_for(threads, _ints(args.tp), _ints(args.offsets))
    lo, hi = _ints(args.seed_range)
    res = search.search(observed, rows, tensor.shape, args.dist, hyps, seed_range=(lo, hi),
                        budget_seconds=args.search_budget, backend_name=args.backend, n=64, max_miss=6,
                        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                        progress=lambda a, b, h, n: print(f"{b / h:6.1%} {n} hits", file=sys.stderr))
    print(dumps(res))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gpu-forensics", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="read-only feasibility check of a Hugging Face model")
    p.add_argument("model", help="huggingface.co URL, org/name or org/name@revision")
    p.add_argument("--revision")
    p.add_argument("--deep", action="store_true", help="read every safetensors header (tensor/dtype census)")
    p.add_argument("--no-tokenizer", action="store_true", help="skip downloading tokenizer files")
    p.add_argument("--output")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("fingerprint", help="sample weights, recover seed and layout, validate")
    p.add_argument("model")
    p.add_argument("--revision")
    p.add_argument("--seed", default="auto",
                   help="auto (known/metadata/conventions, then exhaustive search), screen (no exhaustive "
                        "search), search (exhaustive only), known, or comma-separated seeds")
    p.add_argument("--dist", default="auto", choices=["auto", "normal", "uniform", "trunc_normal"])
    p.add_argument("--tensor", help="tensor name (default: the input embedding)")
    p.add_argument("--rows", help="rows to sample: '1,5,10-20', 'tail:N' or 'random:N' (default: auto-detect)")
    p.add_argument("--max-rows", type=int, default=64)
    p.add_argument("--offsets", default="0", help="Philox offsets to consider, e.g. '0,3776' or '0:8192:4'")
    p.add_argument("--tp", default="1", help="tensor-parallel degrees to consider, e.g. '1,2,4,8'")
    p.add_argument("--search-budget", type=float, default=3600.0, help="seconds per exhaustive search family")
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "numpy", "cuda"],
                   help="seed-search backend (cuda is experimental)")
    p.add_argument("--null-seeds", type=int, default=100)
    p.add_argument("--format", default="json", choices=["json", "text"])
    p.add_argument("--output", help="write the JSON report here")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_fingerprint)

    p = sub.add_parser("validate", help="re-run held-out and null tests for a saved report")
    p.add_argument("report")
    p.add_argument("--null-seeds", type=int, default=200)
    p.add_argument("--fresh-rows", type=int, default=0, help="also test N newly selected rows")
    p.add_argument("--format", default="json", choices=["json", "text"])
    p.add_argument("--output")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("search-seed", help="run the exhaustive seed search directly")
    p.add_argument("model")
    p.add_argument("--revision")
    p.add_argument("--tensor")
    p.add_argument("--rows", required=True)
    p.add_argument("--dist", default="normal", choices=["normal", "uniform", "trunc_normal"])
    p.add_argument("--threads", help="comma-separated G values (default: catalogue)")
    p.add_argument("--tp", default="1")
    p.add_argument("--offsets", default="0")
    p.add_argument("--seed-range", default=f"0,{2**32}")
    p.add_argument("--search-budget", type=float)
    p.add_argument("--checkpoint")
    p.add_argument("--backend", default="auto", choices=["auto", "cpu", "numpy", "cuda"])
    p.set_defaults(func=cmd_search)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except hub.HubError as exc:
        print(f"gpu-forensics: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
