"""Parsing and normalising JSONL usage logs.

A record is one API call. Providers disagree about what "input tokens" means
on the wire: Anthropic's ``input_tokens`` already excludes anything served
from cache, OpenAI's ``prompt_tokens`` includes the cached prefix. Both shapes
are normalised here so everything downstream deals in the same four counts.
"""

import json

__all__ = ["UsageRecord", "parse_record", "load_usage", "aggregate"]


class UsageRecord(object):
    """One normalised usage-log entry."""

    __slots__ = (
        "model",
        "date",
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "raw",
    )

    def __init__(
        self,
        model,
        date,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_write_tokens,
        raw,
    ):
        self.model = model
        self.date = date
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_input_tokens = cached_input_tokens
        self.cache_write_tokens = cache_write_tokens
        self.raw = raw

    def field(self, name):
        """Look up ``model``, ``date``, or any other top-level field of the record."""
        if name == "model":
            return self.model
        if name == "date":
            return self.date
        value = self.raw.get(name)
        return "(unknown)" if value is None else value

    def __repr__(self):
        return "UsageRecord(%s, in=%d, out=%d)" % (
            self.model,
            self.input_tokens,
            self.output_tokens,
        )


def _nonneg_int(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("%s must be a number" % field)
    if value < 0:
        raise ValueError("%s must not be negative" % field)
    return int(value)


def parse_record(obj):
    """Turn one decoded JSON object into a :class:`UsageRecord`.

    Accepts either shape without being told which: Anthropic-style
    ``usage.input_tokens`` / ``usage.output_tokens`` (with optional
    ``cache_read_input_tokens`` / ``cache_creation_input_tokens``), or
    OpenAI-style ``usage.prompt_tokens`` / ``usage.completion_tokens`` (with
    optional ``prompt_tokens_details.cached_tokens``).
    """
    if not isinstance(obj, dict):
        raise ValueError("usage record must be a JSON object")
    model = obj.get("model")
    if not model:
        raise ValueError("usage record is missing 'model'")
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage record is missing a 'usage' object")

    if "input_tokens" in usage or "output_tokens" in usage:
        input_tokens = _nonneg_int(usage.get("input_tokens", 0), "input_tokens")
        output_tokens = _nonneg_int(usage.get("output_tokens", 0), "output_tokens")
        cached_input_tokens = _nonneg_int(
            usage.get("cache_read_input_tokens", 0), "cache_read_input_tokens"
        )
        cache_write_tokens = _nonneg_int(
            usage.get("cache_creation_input_tokens", 0), "cache_creation_input_tokens"
        )
    elif "prompt_tokens" in usage or "completion_tokens" in usage:
        prompt_tokens = _nonneg_int(usage.get("prompt_tokens", 0), "prompt_tokens")
        output_tokens = _nonneg_int(usage.get("completion_tokens", 0), "completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        if not isinstance(details, dict):
            raise ValueError("prompt_tokens_details must be a JSON object")
        cached_input_tokens = _nonneg_int(details.get("cached_tokens", 0), "cached_tokens")
        if cached_input_tokens > prompt_tokens:
            raise ValueError("cached_tokens exceeds prompt_tokens")
        input_tokens = prompt_tokens - cached_input_tokens
        cache_write_tokens = 0
    else:
        raise ValueError(
            "usage object has neither Anthropic- nor OpenAI-shaped token fields"
        )

    return UsageRecord(
        model=model,
        date=obj.get("date", ""),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        raw=obj,
    )


def load_usage(text, strict=False):
    """Parse a JSONL usage log.

    Returns ``(records, problems)``. Blank lines are skipped. A malformed
    line is described in ``problems`` and skipped unless ``strict`` is set,
    in which case the first bad line raises ``ValueError``.
    """
    records = []
    problems = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(parse_record(obj))
        except ValueError as error:
            problem = "line %d: %s" % (line_no, error)
            if strict:
                raise ValueError(problem)
            problems.append(problem)
    return records, problems


def aggregate(records, group_by="model"):
    """Group records by a field name, preserving first-seen order."""
    groups = {}
    for record in records:
        groups.setdefault(record.field(group_by), []).append(record)
    return groups
