# llm-cost

Estimate and report LLM token costs from a price table you control. One command
answers "what will this cost", "what did last week cost", and "what would it
have cost on a cheaper model".

Token pricing is simple arithmetic that is easy to get wrong in ways that only
show up on the invoice: cached tokens billed at the full input rate, cache
writes ignored entirely, an unpriced experimental model quietly summing to zero.
`llm-cost` handles those cases explicitly and shows its work.

- **Four token classes**, priced separately: input, output, cache reads, cache
  writes.
- **Reads OpenAI- and Anthropic-shaped usage logs.** OpenAI's `prompt_tokens`
  includes the cached prefix and Anthropic's `input_tokens` does not; both are
  normalised before anything is multiplied by a price.
- **Never silently drops a model.** Records naming an unpriced model are counted
  and reported, not folded into the total as zero.
- **Bring your own prices.** The built-in table is a dated snapshot; a JSON file
  overrides any part of it, or replaces it outright.

Pure Python, standard library only, Python 3.9+.

## Install

```bash
pip install .
```

Or run it straight out of a checkout:

```bash
python -m llm_cost.cli estimate --model claude-opus-5 --input 12000 --output 800
```

## Usage

### Estimate one call

```console
$ llm-cost estimate --model claude-opus-5 --input 12000 --output 800
claude-opus-5  (1 call, prices as of 2026-06-24)

item          tokens   $/1M     cost
------------  ------  -----  -------
input         12,000   5.00  $0.0600
cached input       0   0.50  $0.0000
cache write        0   6.25  $0.0000
output           800  25.00  $0.0200
total         12,800         $0.0800

cost per call: $0.0800
```

Add `--calls 50000` to size a batch job, and `--cached` / `--cache-write` when
the workload reuses a cached prefix.

### Report on a usage log

The input is JSONL, one API call per line, in whatever shape your SDK logs:

```json
{"model":"claude-opus-5","date":"2026-06-01T09:12:00Z","team":"agents","usage":{"input_tokens":18400,"output_tokens":2100,"cache_read_input_tokens":52000,"cache_creation_input_tokens":9000}}
{"model":"gpt-4o-mini","date":"2026-06-01T10:03:00Z","team":"search","usage":{"prompt_tokens":31000,"completion_tokens":420,"prompt_tokens_details":{"cached_tokens":24000}}}
```

```console
$ llm-cost report usage.jsonl --group-by model
model             calls    input   cached  output     cost   $/call
----------------  -----  -------  -------  ------  -------  -------
claude-opus-5         2   27,600  104,000   3,500  $0.3337  $0.1669
claude-haiku-4-5      1  140,000        0   9,800  $0.1890  $0.1890
gemini-2.5-flash      1   88,000        0   3,100  $0.0341  $0.0341
gpt-4o-mini           1    7,000   24,000     420  $0.0031  $0.0031
TOTAL                 5  262,600  128,000  16,820  $0.5600

skipped 1 record(s) with no price: internal-router-v3
```

`--group-by` takes `model`, `date`, or any other top-level field in your log.
`date` buckets by calendar day even when the log carries full timestamps, as
in the example above (`2026-06-01T09:12:00Z` groups under `2026-06-01`):

```console
$ llm-cost report usage.jsonl --group-by team
team    calls    input   cached  output     cost   $/call
------  -----  -------  -------  ------  -------  -------
agents      2   27,600  104,000   3,500  $0.3337  $0.1669
search      2  147,000   24,000  10,220  $0.1921  $0.0961
ops         1   88,000        0   3,100  $0.0341  $0.0341
TOTAL       5  262,600  128,000  16,820  $0.5600
```

Malformed lines are listed and skipped so one bad row cannot lose a month of
data. Use `--strict` in CI to fail instead.

### Compare models

```console
$ llm-cost compare --input 10000 --output 1000 --provider anthropic
10,000 in + 1,000 out, 1 call, prices as of 2026-06-24

model              provider   $/1M in  $/1M out     cost  vs cheapest
-----------------  ---------  -------  --------  -------  -----------
claude-haiku-4-5   anthropic     1.00      5.00  $0.0150         1.0x
claude-sonnet-4-6  anthropic     3.00     15.00  $0.0450         3.0x
claude-sonnet-5    anthropic     3.00     15.00  $0.0450         3.0x
claude-opus-4-6    anthropic     5.00     25.00  $0.0750         5.0x
claude-opus-4-7    anthropic     5.00     25.00  $0.0750         5.0x
claude-opus-4-8    anthropic     5.00     25.00  $0.0750         5.0x
claude-opus-5      anthropic     5.00     25.00  $0.0750         5.0x
claude-fable-5     anthropic    10.00     50.00  $0.1500        10.0x
```

`--models a,b,c` narrows the comparison to a shortlist.

### Inspect the table

```console
$ llm-cost models
13 models, USD per 1M tokens, as of 2026-06-24 (source: built-in)
...
```

Every subcommand accepts `--json` for machine-readable output and `--pricing`
for an override file, before or after the subcommand.

## Prices are a snapshot, not a source of truth

The built-in table holds public list prices captured on **2026-06-24**. It will
drift, it does not know about your negotiated rates, discounts, batch pricing or
regional surcharges, and it covers only a handful of models. Override it:

```json
{
  "as_of": "2026-07-01",
  "replace": false,
  "models": {
    "claude-opus-5": { "input": 4.25, "output": 21.0, "cached_input": 0.42, "cache_write": 5.3 },
    "internal-router-v3": { "input": 0.2, "output": 0.8, "provider": "internal" }
  }
}
```

```bash
llm-cost --pricing prices.json report usage.jsonl
```

Entries merge onto the built-in table; set `"replace": true` to start from an
empty table so an unlisted model raises instead of pricing against a stale
default. `cached_input` and `cache_write` fall back to the input price when
omitted, which over-states rather than under-states the bill.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `2` | Bad input: unreadable file, invalid JSON, invalid argument |
| `3` | The named model has no price in the table |

## Library API

```python
from llm_cost import default_pricing, load_pricing, estimate_cost, load_usage, build_report

table = load_pricing("prices.json")        # or default_pricing()
price = table.resolve("anthropic.claude-opus-5-20260101")   # prefixes and suffixes tolerated

one = estimate_cost(price, input_tokens=12000, output_tokens=800, cached_input_tokens=52000)
one.total_cost          # 0.106
one.cost_per_call
one.to_dict()

records, problems = load_usage(open("usage.jsonl").read())
report = build_report(records, table, group_by="team")
report.total_cost
report.unknown_models   # {'internal-router-v3'}
```

| Function | Purpose |
| --- | --- |
| `default_pricing()` / `load_pricing(path)` / `parse_pricing(obj)` | Build a `PricingTable` |
| `PricingTable.resolve(name)` | Name resolution, raises `UnknownModelError` |
| `estimate_cost(price, ...)` | Cost of one call or `calls` identical calls |
| `load_usage(text, strict=False)` | Parse JSONL into records plus problems |
| `build_report(records, table, group_by=...)` | Aggregate and cost a log |
| `compare_models(table, input, output, ...)` | Rank models cheapest first |
| `render_table(headers, rows)` | The aligned table renderer |

## Test

```bash
python -m pytest tests -q
```

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 chris-h116.
