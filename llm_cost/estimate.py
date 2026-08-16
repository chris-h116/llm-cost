"""Cost arithmetic.

One convention runs through the whole package: ``input_tokens`` never includes
cached tokens. Providers disagree about this on the wire -- OpenAI's
``prompt_tokens`` is inclusive, Anthropic's ``input_tokens`` is exclusive -- so
:mod:`llm_cost.usage` normalises to the exclusive form before anything gets
multiplied by a price.
"""

__all__ = ["CostBreakdown", "estimate_cost", "PER_MILLION"]

PER_MILLION = 1000000.0


class CostBreakdown(object):
    """Cost of one call, or of an aggregate of calls, split by token class."""

    __slots__ = (
        "model",
        "calls",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "input_cost",
        "output_cost",
        "cached_input_cost",
        "cache_write_cost",
    )

    def __init__(
        self,
        model,
        calls,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        input_cost,
        output_cost,
        cached_input_cost,
        cache_write_cost,
    ):
        self.model = model
        self.calls = calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.input_cost = input_cost
        self.output_cost = output_cost
        self.cached_input_cost = cached_input_cost
        self.cache_write_cost = cache_write_cost

    @property
    def total_tokens(self):
        return (
            self.input_tokens
            + self.output_tokens
            + self.cached_input_tokens
            + self.cache_write_tokens
        )

    @property
    def total_cost(self):
        return (
            self.input_cost
            + self.output_cost
            + self.cached_input_cost
            + self.cache_write_cost
        )

    @property
    def cost_per_call(self):
        if not self.calls:
            return 0.0
        return self.total_cost / self.calls

    def to_dict(self):
        return {
            "model": self.model,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "cached_input_cost": round(self.cached_input_cost, 6),
            "cache_write_cost": round(self.cache_write_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "cost_per_call": round(self.cost_per_call, 6),
        }

    def __repr__(self):
        return "CostBreakdown(%s, calls=%d, total=%.6f)" % (
            self.model,
            self.calls,
            self.total_cost,
        )


def estimate_cost(
    price,
    input_tokens=0,
    output_tokens=0,
    cached_input_tokens=0,
    cache_write_tokens=0,
    calls=1,
):
    """Price one call (or ``calls`` identical calls) against ``price``.

    Token counts are per call and must not be negative. ``input_tokens`` is the
    uncached remainder: pass cache reads as ``cached_input_tokens`` instead of
    counting them twice.
    """
    counts = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "calls": calls,
    }
    for name, value in counts.items():
        if not isinstance(value, int):
            raise ValueError("%s must be an integer, got %r" % (name, value))
        if value < 0:
            raise ValueError("%s must not be negative" % name)

    scale = calls / PER_MILLION
    return CostBreakdown(
        model=price.model,
        calls=calls,
        input_tokens=input_tokens * calls,
        output_tokens=output_tokens * calls,
        cached_input_tokens=cached_input_tokens * calls,
        cache_write_tokens=cache_write_tokens * calls,
        input_cost=input_tokens * price.input * scale,
        output_cost=output_tokens * price.output * scale,
        cached_input_cost=cached_input_tokens * price.cached_input * scale,
        cache_write_cost=cache_write_tokens * price.cache_write * scale,
    )
