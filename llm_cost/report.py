"""Aggregating priced usage into a report, and ranking models against each other."""

from .estimate import estimate_cost
from .pricing import UnknownModelError

__all__ = ["GroupSummary", "Report", "build_report", "compare_models"]


class GroupSummary(object):
    """Totals for one group (one model, one team, one day, ...)."""

    __slots__ = (
        "key",
        "calls",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "cache_write_tokens",
        "cost",
    )

    def __init__(
        self, key, calls, input_tokens, cached_input_tokens, output_tokens, cache_write_tokens, cost
    ):
        self.key = key
        self.calls = calls
        self.input_tokens = input_tokens
        self.cached_input_tokens = cached_input_tokens
        self.output_tokens = output_tokens
        self.cache_write_tokens = cache_write_tokens
        self.cost = cost

    @property
    def cost_per_call(self):
        if not self.calls:
            return 0.0
        return self.cost / self.calls

    def to_dict(self):
        return {
            "key": self.key,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cost": round(self.cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }


class Report(object):
    """A usage log aggregated into groups, cheapest cost ordering aside."""

    __slots__ = ("group_by", "groups", "unknown_models", "skipped_records")

    def __init__(self, group_by, groups, unknown_models, skipped_records):
        self.group_by = group_by
        self.groups = groups
        self.unknown_models = unknown_models
        self.skipped_records = skipped_records

    @property
    def total_calls(self):
        return sum(group.calls for group in self.groups)

    @property
    def total_input_tokens(self):
        return sum(group.input_tokens for group in self.groups)

    @property
    def total_cached_input_tokens(self):
        return sum(group.cached_input_tokens for group in self.groups)

    @property
    def total_output_tokens(self):
        return sum(group.output_tokens for group in self.groups)

    @property
    def total_cache_write_tokens(self):
        return sum(group.cache_write_tokens for group in self.groups)

    @property
    def total_cost(self):
        return sum(group.cost for group in self.groups)

    def to_dict(self):
        return {
            "group_by": self.group_by,
            "groups": [group.to_dict() for group in self.groups],
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_cached_input_tokens": self.total_cached_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_write_tokens": self.total_cache_write_tokens,
            "total_cost": round(self.total_cost, 6),
            "skipped_records": self.skipped_records,
            "unknown_models": sorted(self.unknown_models),
        }


def build_report(records, table, group_by="model"):
    """Price every record against ``table`` and group the results.

    Groups are sorted most expensive first. A record naming a model that has
    no price is counted in ``unknown_models`` / ``skipped_records`` rather
    than silently costing zero.
    """
    totals = {}
    order = []
    unknown_models = set()
    skipped_records = 0

    for record in records:
        try:
            price = table.resolve(record.model)
        except UnknownModelError:
            unknown_models.add(record.model)
            skipped_records += 1
            continue

        breakdown = estimate_cost(
            price,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_input_tokens,
            cache_write_tokens=record.cache_write_tokens,
        )

        key = record.field(group_by)
        if key not in totals:
            totals[key] = [0, 0, 0, 0, 0, 0.0]
            order.append(key)
        running = totals[key]
        running[0] += 1
        running[1] += breakdown.input_tokens
        running[2] += breakdown.cached_input_tokens
        running[3] += breakdown.output_tokens
        running[4] += breakdown.cache_write_tokens
        running[5] += breakdown.total_cost

    groups = [GroupSummary(key, *totals[key]) for key in order]
    groups.sort(key=lambda group: group.cost, reverse=True)
    return Report(group_by, groups, unknown_models, skipped_records)


def compare_models(table, input_tokens, output_tokens, calls=1, provider=None, models=None):
    """Price the same call against every model and rank cheapest first.

    ``models`` narrows the comparison to a shortlist of names (each resolved
    against ``table``, so an unknown name raises ``UnknownModelError``);
    ``provider`` filters the full table down to one provider.
    """
    names = list(models) if models is not None else table.models()
    ranked = []
    for name in names:
        price = table.resolve(name)
        if provider and price.provider != provider:
            continue
        breakdown = estimate_cost(
            price, input_tokens=input_tokens, output_tokens=output_tokens, calls=calls
        )
        ranked.append((price, breakdown))
    ranked.sort(key=lambda pair: pair[1].total_cost)
    return ranked
