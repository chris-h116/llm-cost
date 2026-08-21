import unittest

from llm_cost.pricing import ModelPrice, PricingTable, UnknownModelError
from llm_cost.report import build_report, compare_models
from llm_cost.usage import UsageRecord


def _table():
    return PricingTable(
        {
            "model-a": ModelPrice(
                "model-a", input=1.0, output=2.0, cached_input=0.5, cache_write=1.5, provider="test"
            ),
            "model-b": ModelPrice(
                "model-b", input=10.0, output=20.0, cached_input=5.0, cache_write=15.0, provider="test"
            ),
        },
        as_of="test",
        source="test",
    )


def _record(
    model, input_tokens, output_tokens, cached_input_tokens=0, cache_write_tokens=0, raw=None, date="2026-01-01"
):
    return UsageRecord(
        model=model,
        date=date,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        raw=raw or {"model": model},
    )


class BuildReport(unittest.TestCase):
    def test_prices_and_totals_one_group(self):
        records = [
            _record("model-a", 1000000, 1000000),
            _record("model-a", 1000000, 1000000),
        ]
        report = build_report(records, _table(), group_by="model")
        self.assertEqual(len(report.groups), 1)
        group = report.groups[0]
        self.assertEqual(group.key, "model-a")
        self.assertEqual(group.calls, 2)
        self.assertAlmostEqual(group.cost, 6.0)  # 2 calls * (1.0 + 2.0)
        self.assertAlmostEqual(group.cost_per_call, 3.0)
        self.assertEqual(report.total_calls, 2)
        self.assertAlmostEqual(report.total_cost, 6.0)

    def test_unknown_model_is_reported_not_zeroed(self):
        records = [_record("model-a", 1000000, 0), _record("unpriced-model", 1000000, 1000000)]
        report = build_report(records, _table(), group_by="model")
        self.assertEqual(report.unknown_models, {"unpriced-model"})
        self.assertEqual(report.skipped_records, 1)
        self.assertEqual(len(report.groups), 1)
        self.assertEqual(report.groups[0].key, "model-a")

    def test_groups_sort_most_expensive_first(self):
        records = [
            _record("model-a", 1000000, 0),  # cost 1.0
            _record("model-b", 1000000, 0),  # cost 10.0
        ]
        report = build_report(records, _table(), group_by="model")
        self.assertEqual([group.key for group in report.groups], ["model-b", "model-a"])

    def test_group_by_arbitrary_field(self):
        records = [
            _record("model-a", 1000000, 0, raw={"model": "model-a", "team": "agents"}),
            _record("model-b", 1000000, 0, raw={"model": "model-b", "team": "agents"}),
        ]
        report = build_report(records, _table(), group_by="team")
        self.assertEqual(len(report.groups), 1)
        self.assertEqual(report.groups[0].key, "agents")
        self.assertEqual(report.groups[0].calls, 2)

    def test_group_by_date_buckets_timestamps_by_calendar_day(self):
        records = [
            _record("model-a", 1000000, 0, date="2026-06-01T09:12:00Z"),
            _record("model-a", 1000000, 0, date="2026-06-01T22:45:00Z"),
            _record("model-a", 1000000, 0, date="2026-06-02T00:01:00Z"),
        ]
        report = build_report(records, _table(), group_by="date")
        self.assertEqual(
            sorted(group.key for group in report.groups), ["2026-06-01", "2026-06-02"]
        )
        by_day = dict((group.key, group.calls) for group in report.groups)
        self.assertEqual(by_day["2026-06-01"], 2)
        self.assertEqual(by_day["2026-06-02"], 1)

    def test_cost_per_call_is_zero_for_empty_group(self):
        report = build_report([], _table(), group_by="model")
        self.assertEqual(report.groups, [])
        self.assertEqual(report.total_calls, 0)
        self.assertEqual(report.total_cost, 0)


class CompareModels(unittest.TestCase):
    def test_ranked_cheapest_first(self):
        ranked = compare_models(_table(), input_tokens=1000, output_tokens=500)
        self.assertEqual([price.model for price, _ in ranked], ["model-a", "model-b"])
        self.assertLess(ranked[0][1].total_cost, ranked[1][1].total_cost)

    def test_provider_filter(self):
        table = _table()
        table.prices["model-c"] = ModelPrice(
            "model-c", input=0.1, output=0.1, provider="other"
        )
        ranked = compare_models(table, input_tokens=1000, output_tokens=500, provider="test")
        self.assertEqual({price.model for price, _ in ranked}, {"model-a", "model-b"})

    def test_models_shortlist(self):
        ranked = compare_models(
            _table(), input_tokens=1000, output_tokens=500, models=["model-b"]
        )
        self.assertEqual([price.model for price, _ in ranked], ["model-b"])

    def test_unknown_model_in_shortlist_raises(self):
        with self.assertRaises(UnknownModelError):
            compare_models(_table(), input_tokens=1000, output_tokens=500, models=["nope"])


if __name__ == "__main__":
    unittest.main()
