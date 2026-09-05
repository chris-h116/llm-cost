"""Exercises the public `llm_cost` namespace the README's Library API section documents.

Every other test file imports from a submodule (llm_cost.pricing, llm_cost.usage,
...); none of them catch a name missing from the top-level package or the pieces
failing to interoperate when wired together the way the README shows.
"""

import unittest

from llm_cost import (
    ModelPrice,
    PricingTable,
    UnknownModelError,
    build_report,
    compare_models,
    default_pricing,
    estimate_cost,
    load_usage,
    render_table,
)


class LibraryApiWalkthrough(unittest.TestCase):
    """Mirrors the README's "Library API" example line for line."""

    def test_resolve_estimate_and_report_match_the_documented_values(self):
        table = default_pricing()
        price = table.resolve("anthropic.claude-opus-5-20260101")
        self.assertEqual(price.model, "claude-opus-5")

        one = estimate_cost(
            price, input_tokens=12000, output_tokens=800, cached_input_tokens=52000
        )
        self.assertAlmostEqual(one.total_cost, 0.106)
        self.assertAlmostEqual(one.cost_per_call, 0.106)
        self.assertEqual(one.to_dict()["total_cost"], round(0.106, 6))

        log = (
            '{"model":"claude-opus-5","team":"agents",'
            '"usage":{"input_tokens":12000,"output_tokens":800,'
            '"cache_read_input_tokens":52000}}\n'
            '{"model":"internal-router-v3","team":"agents",'
            '"usage":{"input_tokens":1,"output_tokens":1}}\n'
        )
        records, problems = load_usage(log)
        self.assertEqual(problems, [])

        report = build_report(records, table, group_by="team")
        self.assertAlmostEqual(report.total_cost, one.total_cost)
        self.assertEqual(report.unknown_models, {"internal-router-v3"})

    def test_compare_and_render_table_are_reachable_from_the_top_level(self):
        table = default_pricing()
        ranked = compare_models(table, input_tokens=1000, output_tokens=500, provider="anthropic")
        rows = [[price.model, "%.2f" % breakdown.total_cost] for price, breakdown in ranked]
        rendered = render_table(["model", "cost"], rows)
        self.assertIn(ranked[0][0].model, rendered)

    def test_pricing_types_and_error_are_reachable_from_the_top_level(self):
        table = PricingTable({"m": ModelPrice("m", input=1.0, output=2.0)}, as_of="test")
        with self.assertRaises(UnknownModelError):
            table.resolve("not-in-table")


if __name__ == "__main__":
    unittest.main()
