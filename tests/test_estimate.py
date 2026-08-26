import unittest

from llm_cost.estimate import CostBreakdown, estimate_cost
from llm_cost.pricing import ModelPrice


def _price():
    return ModelPrice(
        "m", input=1.0, output=2.0, cached_input=0.5, cache_write=1.25
    )


class EstimateCost(unittest.TestCase):
    def test_single_call_costs_are_scaled_per_million(self):
        breakdown = estimate_cost(_price(), input_tokens=1000000, output_tokens=500000)
        self.assertEqual(breakdown.input_cost, 1.0)
        self.assertEqual(breakdown.output_cost, 1.0)
        self.assertEqual(breakdown.total_cost, 2.0)

    def test_cached_input_and_cache_write_use_their_own_prices(self):
        breakdown = estimate_cost(
            _price(),
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=1000000,
            cache_write_tokens=1000000,
        )
        self.assertEqual(breakdown.cached_input_cost, 0.5)
        self.assertEqual(breakdown.cache_write_cost, 1.25)

    def test_calls_multiplies_both_tokens_and_cost(self):
        breakdown = estimate_cost(_price(), input_tokens=100, output_tokens=100, calls=3)
        self.assertEqual(breakdown.calls, 3)
        self.assertEqual(breakdown.input_tokens, 300)
        self.assertEqual(breakdown.output_tokens, 300)
        self.assertAlmostEqual(breakdown.total_cost, 3 * (100 * 1.0 + 100 * 2.0) / 1000000.0)

    def test_model_is_carried_from_the_price(self):
        breakdown = estimate_cost(_price())
        self.assertEqual(breakdown.model, "m")

    def test_non_integer_token_count_raises(self):
        with self.assertRaises(ValueError):
            estimate_cost(_price(), input_tokens=1.5)

    def test_negative_token_count_raises(self):
        with self.assertRaises(ValueError):
            estimate_cost(_price(), input_tokens=-1)

    def test_negative_calls_raises(self):
        with self.assertRaises(ValueError):
            estimate_cost(_price(), calls=-1)


class CostBreakdownProperties(unittest.TestCase):
    def test_total_tokens_sums_every_class(self):
        breakdown = estimate_cost(
            _price(),
            input_tokens=1,
            output_tokens=2,
            cached_input_tokens=3,
            cache_write_tokens=4,
        )
        self.assertEqual(breakdown.total_tokens, 10)

    def test_cost_per_call_divides_total_cost_by_calls(self):
        breakdown = estimate_cost(_price(), input_tokens=1000000, calls=4)
        self.assertEqual(breakdown.total_cost, 4.0)
        self.assertEqual(breakdown.cost_per_call, 1.0)

    def test_cost_per_call_is_zero_when_there_are_no_calls(self):
        breakdown = CostBreakdown(
            model="m",
            calls=0,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            cache_write_tokens=0,
            input_cost=0.0,
            output_cost=0.0,
            cached_input_cost=0.0,
            cache_write_cost=0.0,
        )
        self.assertEqual(breakdown.cost_per_call, 0.0)

    def test_to_dict_rounds_costs_to_six_places(self):
        breakdown = estimate_cost(_price(), input_tokens=1)
        as_dict = breakdown.to_dict()
        self.assertEqual(as_dict["input_cost"], round(1 * 1.0 / 1000000.0, 6))
        self.assertEqual(as_dict["model"], "m")
        self.assertIn("total_cost", as_dict)
        self.assertIn("cost_per_call", as_dict)

    def test_repr_includes_model_and_total(self):
        breakdown = estimate_cost(_price(), input_tokens=1000000)
        self.assertIn("m", repr(breakdown))
        self.assertIn("calls=1", repr(breakdown))


if __name__ == "__main__":
    unittest.main()
