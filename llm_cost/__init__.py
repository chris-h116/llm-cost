"""Estimate and report LLM token costs from an overridable price table."""

from .estimate import CostBreakdown, estimate_cost
from .pricing import (
    BUILTIN_PRICING,
    PRICING_AS_OF,
    PRICING_NOTE,
    ModelPrice,
    PricingTable,
    UnknownModelError,
    default_pricing,
    load_pricing,
    parse_pricing,
)
from .report import GroupSummary, Report, build_report, compare_models
from .table import format_int, format_money, render_table
from .usage import UsageRecord, aggregate, load_usage, parse_record

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_PRICING",
    "CostBreakdown",
    "GroupSummary",
    "ModelPrice",
    "PRICING_AS_OF",
    "PRICING_NOTE",
    "PricingTable",
    "Report",
    "UnknownModelError",
    "UsageRecord",
    "__version__",
    "aggregate",
    "build_report",
    "compare_models",
    "default_pricing",
    "estimate_cost",
    "format_int",
    "format_money",
    "load_pricing",
    "load_usage",
    "parse_pricing",
    "parse_record",
    "render_table",
]
