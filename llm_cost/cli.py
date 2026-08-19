"""Command-line entry point: estimate / report / compare / models."""

import argparse
import json
import sys

from .estimate import estimate_cost
from .pricing import UnknownModelError, default_pricing, load_pricing
from .report import build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import load_usage

__all__ = ["main"]


def _build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pricing", metavar="FILE", help="JSON pricing override file")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    parser = argparse.ArgumentParser(prog="llm-cost", parents=[common])
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser(
        "estimate", parents=[common], help="cost of one call or a batch of identical calls"
    )
    estimate_parser.add_argument("--model", required=True)
    estimate_parser.add_argument("--input", type=int, default=0)
    estimate_parser.add_argument("--output", type=int, default=0)
    estimate_parser.add_argument("--cached", type=int, default=0, help="cached input tokens")
    estimate_parser.add_argument("--cache-write", type=int, default=0)
    estimate_parser.add_argument("--calls", type=int, default=1)

    report_parser = subparsers.add_parser(
        "report", parents=[common], help="cost a JSONL usage log"
    )
    report_parser.add_argument("path")
    report_parser.add_argument("--group-by", default="model")
    report_parser.add_argument(
        "--strict", action="store_true", help="fail on the first malformed line"
    )

    compare_parser = subparsers.add_parser(
        "compare", parents=[common], help="rank models by cost for one shaped call"
    )
    compare_parser.add_argument("--input", type=int, default=0)
    compare_parser.add_argument("--output", type=int, default=0)
    compare_parser.add_argument("--calls", type=int, default=1)
    compare_parser.add_argument("--provider")
    compare_parser.add_argument("--models", help="comma-separated shortlist of model names")

    subparsers.add_parser("models", parents=[common], help="list the resolved price table")

    return parser


def _load_table(args):
    if args.pricing:
        return load_pricing(args.pricing)
    return default_pricing()


def _cmd_estimate(args, table):
    price = table.resolve(args.model)
    breakdown = estimate_cost(
        price,
        input_tokens=args.input,
        output_tokens=args.output,
        cached_input_tokens=args.cached,
        cache_write_tokens=args.cache_write,
        calls=args.calls,
    )

    if args.json:
        return json.dumps(breakdown.to_dict(), indent=2)

    headers = ["item", "tokens", "$/1M", "cost"]
    rows = [
        ["input", format_int(breakdown.input_tokens), "%.2f" % price.input, format_money(breakdown.input_cost)],
        [
            "cached input",
            format_int(breakdown.cached_input_tokens),
            "%.2f" % price.cached_input,
            format_money(breakdown.cached_input_cost),
        ],
        [
            "cache write",
            format_int(breakdown.cache_write_tokens),
            "%.2f" % price.cache_write,
            format_money(breakdown.cache_write_cost),
        ],
        ["output", format_int(breakdown.output_tokens), "%.2f" % price.output, format_money(breakdown.output_cost)],
        ["total", format_int(breakdown.total_tokens), "", format_money(breakdown.total_cost)],
    ]
    lines = [
        "%s  (%d call%s, prices as of %s)"
        % (price.model, args.calls, "" if args.calls == 1 else "s", table.as_of),
        "",
        render_table(headers, rows),
        "",
        "cost per call: %s" % format_money(breakdown.cost_per_call),
    ]
    return "\n".join(lines)


def _cmd_report(args, table):
    try:
        with open(args.path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise ValueError("cannot read %s: %s" % (args.path, error))

    records, problems = load_usage(text, strict=args.strict)
    report = build_report(records, table, group_by=args.group_by)

    if args.json:
        payload = report.to_dict()
        payload["problems"] = problems
        return json.dumps(payload, indent=2)

    headers = [args.group_by, "calls", "input", "cached", "output", "cost", "$/call"]
    rows = [
        [
            str(group.key),
            format_int(group.calls),
            format_int(group.input_tokens),
            format_int(group.cached_input_tokens),
            format_int(group.output_tokens),
            format_money(group.cost),
            format_money(group.cost_per_call),
        ]
        for group in report.groups
    ]
    rows.append(
        [
            "TOTAL",
            format_int(report.total_calls),
            format_int(report.total_input_tokens),
            format_int(report.total_cached_input_tokens),
            format_int(report.total_output_tokens),
            format_money(report.total_cost),
            "",
        ]
    )

    lines = [render_table(headers, rows)]
    if report.skipped_records:
        lines.append("")
        lines.append(
            "skipped %d record(s) with no price: %s"
            % (report.skipped_records, ", ".join(sorted(report.unknown_models)))
        )
    if problems:
        lines.append("")
        lines.append("skipped %d malformed line(s):" % len(problems))
        lines.extend("  " + problem for problem in problems)
    return "\n".join(lines)


def _cmd_compare(args, table):
    models = None
    if args.models:
        models = [name.strip() for name in args.models.split(",") if name.strip()]

    ranked = compare_models(
        table,
        input_tokens=args.input,
        output_tokens=args.output,
        calls=args.calls,
        provider=args.provider,
        models=models,
    )
    if not ranked:
        raise ValueError("no models matched the given filters")

    if args.json:
        payload = [
            dict(
                breakdown.to_dict(),
                provider=price.provider,
                input_price=price.input,
                output_price=price.output,
            )
            for price, breakdown in ranked
        ]
        return json.dumps(payload, indent=2)

    cheapest = ranked[0][1].total_cost
    headers = ["model", "provider", "$/1M in", "$/1M out", "cost", "vs cheapest"]
    rows = []
    for price, breakdown in ranked:
        ratio = breakdown.total_cost / cheapest if cheapest else 0.0
        rows.append(
            [
                price.model,
                price.provider,
                "%.2f" % price.input,
                "%.2f" % price.output,
                format_money(breakdown.total_cost),
                "%.1fx" % ratio,
            ]
        )

    lines = [
        "%s in + %s out, %d call%s, prices as of %s"
        % (
            format_int(args.input),
            format_int(args.output),
            args.calls,
            "" if args.calls == 1 else "s",
            table.as_of,
        ),
        "",
        render_table(headers, rows),
    ]
    return "\n".join(lines)


def _cmd_models(args, table):
    if args.json:
        payload = {
            "as_of": table.as_of,
            "source": table.source,
            "models": dict((name, table.resolve(name).to_dict()) for name in table.models()),
        }
        return json.dumps(payload, indent=2)

    headers = ["model", "provider", "$/1M in", "$/1M out", "$/1M cached", "$/1M write"]
    rows = []
    for name in table.models():
        price = table.resolve(name)
        rows.append(
            [
                name,
                price.provider,
                "%.2f" % price.input,
                "%.2f" % price.output,
                "%.2f" % price.cached_input,
                "%.2f" % price.cache_write,
            ]
        )

    lines = [
        "%d models, USD per 1M tokens, as of %s (source: %s)" % (len(table), table.as_of, table.source),
        "",
        render_table(headers, rows),
    ]
    return "\n".join(lines)


_HANDLERS = {
    "estimate": _cmd_estimate,
    "report": _cmd_report,
    "compare": _cmd_compare,
    "models": _cmd_models,
}


def main(argv=None):
    args = _build_parser().parse_args(argv)
    try:
        table = _load_table(args)
        output = _HANDLERS[args.command](args, table)
    except UnknownModelError as error:
        sys.stderr.write("llm-cost: %s\n" % error)
        return 3
    except ValueError as error:
        sys.stderr.write("llm-cost: %s\n" % error)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
