"""The price table, its overrides, and model-name resolution.

Prices are USD per one million tokens. The built-in table is a *snapshot*: list
prices move, contract prices differ from list prices, and no library can keep up
with either. Treat it as a starting point and override it with ``--pricing``
once the numbers matter.
"""

import json
import os

__all__ = [
    "ModelPrice",
    "PricingTable",
    "UnknownModelError",
    "BUILTIN_PRICING",
    "PRICING_AS_OF",
    "PRICING_NOTE",
    "load_pricing",
    "default_pricing",
]

#: Date the built-in numbers were taken from the vendors' published list prices.
PRICING_AS_OF = "2026-06-24"

PRICING_NOTE = (
    "Built-in prices are public list prices captured on %s, in USD per 1M "
    "tokens. Verify them against your own invoice and override with "
    "--pricing FILE." % PRICING_AS_OF
)


class UnknownModelError(KeyError):
    """Raised when a model name cannot be resolved against the price table."""

    def __str__(self):
        # KeyError.__str__ reprs its argument, which puts stray quotes around
        # every message printed by the CLI.
        return self.args[0] if self.args else ""


class ModelPrice(object):
    """Per-million-token prices for one model."""

    __slots__ = ("model", "provider", "input", "output", "cached_input", "cache_write")

    def __init__(
        self, model, input, output, cached_input=None, cache_write=None, provider=""
    ):
        self.model = model
        self.provider = provider
        self.input = float(input)
        self.output = float(output)
        # A cache read defaults to the full input price: charging less than the
        # vendor does would quietly understate every report.
        self.cached_input = self.input if cached_input is None else float(cached_input)
        self.cache_write = self.input if cache_write is None else float(cache_write)
        for field in ("input", "output", "cached_input", "cache_write"):
            if getattr(self, field) < 0:
                raise ValueError(
                    "%s: %s price must not be negative" % (model, field)
                )

    def to_dict(self):
        return {
            "provider": self.provider,
            "input": self.input,
            "output": self.output,
            "cached_input": self.cached_input,
            "cache_write": self.cache_write,
        }

    def __repr__(self):
        return "ModelPrice(%s, in=%.4f, out=%.4f)" % (
            self.model,
            self.input,
            self.output,
        )


def _price(model, provider, input, output, cached_input, cache_write):
    return ModelPrice(
        model=model,
        provider=provider,
        input=input,
        output=output,
        cached_input=cached_input,
        cache_write=cache_write,
    )


#: model -> ModelPrice. USD per 1M tokens, list prices as of PRICING_AS_OF.
BUILTIN_PRICING = dict(
    (price.model, price)
    for price in [
        # Anthropic: cache reads bill at 0.1x input, 5-minute cache writes at 1.25x.
        _price("claude-fable-5", "anthropic", 10.00, 50.00, 1.00, 12.50),
        _price("claude-opus-5", "anthropic", 5.00, 25.00, 0.50, 6.25),
        _price("claude-opus-4-8", "anthropic", 5.00, 25.00, 0.50, 6.25),
        _price("claude-opus-4-7", "anthropic", 5.00, 25.00, 0.50, 6.25),
        _price("claude-opus-4-6", "anthropic", 5.00, 25.00, 0.50, 6.25),
        _price("claude-sonnet-5", "anthropic", 3.00, 15.00, 0.30, 3.75),
        _price("claude-sonnet-4-6", "anthropic", 3.00, 15.00, 0.30, 3.75),
        _price("claude-haiku-4-5", "anthropic", 1.00, 5.00, 0.10, 1.25),
        # OpenAI: cached input is discounted, cache writes are not billed extra.
        _price("gpt-4o", "openai", 2.50, 10.00, 1.25, 2.50),
        _price("gpt-4o-mini", "openai", 0.15, 0.60, 0.075, 0.15),
        _price("o3-mini", "openai", 1.10, 4.40, 0.55, 1.10),
        # Google
        _price("gemini-2.5-pro", "google", 1.25, 10.00, 0.3125, 1.25),
        _price("gemini-2.5-flash", "google", 0.30, 2.50, 0.075, 0.30),
    ]
)

#: Provider prefixes seen in usage logs that are not part of the model name.
_PROVIDER_PREFIXES = ("anthropic.", "anthropic/", "openai/", "google/", "models/")


class PricingTable(object):
    """A resolved price table plus the provenance of its numbers."""

    __slots__ = ("prices", "as_of", "source")

    def __init__(self, prices, as_of=PRICING_AS_OF, source="built-in"):
        self.prices = dict(prices)
        self.as_of = as_of
        self.source = source

    def __len__(self):
        return len(self.prices)

    def __contains__(self, model):
        try:
            self.resolve(model)
        except UnknownModelError:
            return False
        return True

    def models(self):
        return sorted(self.prices)

    def resolve(self, model):
        """Resolve a model name, tolerating provider prefixes and date suffixes.

        Usage logs carry names like ``anthropic.claude-opus-5`` or
        ``gpt-4o-2026-01-31``; both should price like the model they name.
        """
        if not model:
            raise UnknownModelError("no model name given")
        name = model.strip()
        if name in self.prices:
            return self.prices[name]
        lowered = name.lower()
        if lowered in self.prices:
            return self.prices[lowered]
        for prefix in _PROVIDER_PREFIXES:
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix) :]
                break
        if lowered in self.prices:
            return self.prices[lowered]
        candidates = [key for key in self.prices if lowered.startswith(key)]
        if candidates:
            return self.prices[max(candidates, key=len)]
        raise UnknownModelError(
            "no price for model %r (known: %s)"
            % (model, ", ".join(self.models()[:5]) + ", ...")
        )


def default_pricing():
    """The built-in table."""
    return PricingTable(BUILTIN_PRICING, PRICING_AS_OF, "built-in")


def _coerce(model, entry):
    if not isinstance(entry, dict):
        raise ValueError("price entry for %r must be an object" % (model,))
    missing = [field for field in ("input", "output") if field not in entry]
    if missing:
        raise ValueError(
            "price entry for %r is missing: %s" % (model, ", ".join(missing))
        )
    for field in ("input", "output", "cached_input", "cache_write"):
        value = entry.get(field)
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(
                "price entry for %r has a non-numeric %s" % (model, field)
            )
    return ModelPrice(
        model=model,
        provider=entry.get("provider", ""),
        input=entry["input"],
        output=entry["output"],
        cached_input=entry.get("cached_input"),
        cache_write=entry.get("cache_write"),
    )


def parse_pricing(data, source="override"):
    """Build a :class:`PricingTable` from a decoded JSON document.

    Two shapes are accepted::

        {"as_of": "2026-07-01", "models": {"my-model": {"input": 1, "output": 3}}}
        {"my-model": {"input": 1, "output": 3}}

    Entries merge onto the built-in table unless ``"replace": true`` is set.
    """
    if not isinstance(data, dict):
        raise ValueError("pricing file must contain a JSON object")
    if "models" in data:
        models = data.get("models")
        as_of = data.get("as_of", "unspecified")
        replace = bool(data.get("replace", False))
    else:
        models = data
        as_of = "unspecified"
        replace = False
    if not isinstance(models, dict):
        raise ValueError("pricing 'models' must be a JSON object")
    if not models:
        raise ValueError("pricing file defines no models")

    prices = {} if replace else dict(BUILTIN_PRICING)
    for model, entry in models.items():
        prices[model] = _coerce(model, entry)
    return PricingTable(prices, as_of, source)


def load_pricing(path):
    """Load a pricing override from ``path``."""
    if not os.path.exists(path):
        raise ValueError("pricing file not found: %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = json.load(handle)
        except ValueError as error:
            raise ValueError("pricing file %s is not valid JSON: %s" % (path, error))
    return parse_pricing(data, source=path)
