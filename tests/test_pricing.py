import os
import tempfile
import unittest

from llm_cost.pricing import (
    BUILTIN_PRICING,
    ModelPrice,
    UnknownModelError,
    default_pricing,
    load_pricing,
    parse_pricing,
)


class ModelPriceDefaults(unittest.TestCase):
    def test_cached_input_and_cache_write_default_to_input_price(self):
        price = ModelPrice("m", input=2.0, output=4.0)
        self.assertEqual(price.cached_input, 2.0)
        self.assertEqual(price.cache_write, 2.0)

    def test_negative_price_is_rejected(self):
        with self.assertRaises(ValueError):
            ModelPrice("m", input=-1.0, output=4.0)


class Resolve(unittest.TestCase):
    def setUp(self):
        self.table = default_pricing()

    def test_exact_match(self):
        self.assertIs(self.table.resolve("claude-opus-5"), BUILTIN_PRICING["claude-opus-5"])

    def test_case_insensitive_match(self):
        self.assertIs(self.table.resolve("Claude-Opus-5"), BUILTIN_PRICING["claude-opus-5"])

    def test_provider_prefix_is_stripped(self):
        self.assertIs(
            self.table.resolve("anthropic.claude-opus-5"), BUILTIN_PRICING["claude-opus-5"]
        )
        self.assertIs(
            self.table.resolve("openai/gpt-4o-mini"), BUILTIN_PRICING["gpt-4o-mini"]
        )

    def test_date_suffix_matches_by_longest_known_prefix(self):
        self.assertIs(
            self.table.resolve("gpt-4o-mini-2026-01-31"), BUILTIN_PRICING["gpt-4o-mini"]
        )

    def test_prefix_match_prefers_the_longest_candidate(self):
        # "gpt-4o" is itself a prefix of "gpt-4o-mini"; a versioned gpt-4o-mini
        # name must resolve to gpt-4o-mini, not the shorter gpt-4o.
        self.assertIs(
            self.table.resolve("gpt-4o-mini-2026-06-01"), BUILTIN_PRICING["gpt-4o-mini"]
        )

    def test_empty_model_name_raises(self):
        with self.assertRaises(UnknownModelError):
            self.table.resolve("")

    def test_unrecognised_model_raises_with_known_models_listed(self):
        with self.assertRaises(UnknownModelError) as raised:
            self.table.resolve("not-a-real-model")
        self.assertIn("not-a-real-model", str(raised.exception))

    def test_contains_does_not_raise_for_unknown_model(self):
        self.assertNotIn("not-a-real-model", self.table)
        self.assertIn("claude-opus-5", self.table)

    def test_mixed_case_override_name_resolves_case_insensitively(self):
        # An override file's keys are whatever casing the author typed; the
        # documented case-insensitive match must hold for those too, not
        # just for the all-lowercase built-in table.
        table = parse_pricing({"My-Internal-Model": {"input": 1, "output": 2}})
        self.assertIs(table.resolve("my-internal-model"), table.resolve("My-Internal-Model"))
        self.assertIs(table.resolve("MY-INTERNAL-MODEL"), table.resolve("my-internal-model"))


class ParsePricing(unittest.TestCase):
    def test_flat_shape_merges_onto_builtin_table(self):
        table = parse_pricing({"my-model": {"input": 1, "output": 3}})
        self.assertIn("my-model", table.prices)
        self.assertIn("claude-opus-5", table.prices)
        self.assertEqual(table.as_of, "unspecified")

    def test_models_shape_carries_as_of_and_merges_by_default(self):
        table = parse_pricing(
            {"as_of": "2026-07-01", "models": {"my-model": {"input": 1, "output": 3}}}
        )
        self.assertEqual(table.as_of, "2026-07-01")
        self.assertIn("claude-opus-5", table.prices)

    def test_replace_true_drops_the_builtin_table(self):
        table = parse_pricing(
            {"models": {"my-model": {"input": 1, "output": 3}}, "replace": True}
        )
        self.assertEqual(list(table.prices), ["my-model"])

    def test_override_can_replace_a_builtin_entry(self):
        table = parse_pricing({"claude-opus-5": {"input": 1, "output": 1}})
        self.assertEqual(table.prices["claude-opus-5"].input, 1.0)

    def test_top_level_not_an_object_raises(self):
        with self.assertRaises(ValueError):
            parse_pricing(["not", "an", "object"])

    def test_models_not_an_object_raises(self):
        with self.assertRaises(ValueError):
            parse_pricing({"models": ["not", "an", "object"]})

    def test_empty_models_raises(self):
        with self.assertRaises(ValueError):
            parse_pricing({"models": {}})

    def test_entry_not_an_object_raises(self):
        with self.assertRaises(ValueError):
            parse_pricing({"my-model": 3})

    def test_entry_missing_input_or_output_raises(self):
        with self.assertRaises(ValueError):
            parse_pricing({"my-model": {"input": 1}})

    def test_entry_with_non_numeric_field_raises(self):
        with self.assertRaises(ValueError):
            parse_pricing({"my-model": {"input": 1, "output": "a lot"}})

    def test_entry_can_omit_optional_cache_fields(self):
        table = parse_pricing({"my-model": {"input": 2, "output": 4}})
        price = table.prices["my-model"]
        self.assertEqual(price.cached_input, 2.0)
        self.assertEqual(price.cache_write, 2.0)


class LoadPricing(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            load_pricing("/no/such/prices.json")

    def test_invalid_json_raises(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        handle.write("not json")
        handle.close()
        try:
            with self.assertRaises(ValueError):
                load_pricing(handle.name)
        finally:
            os.remove(handle.name)

    def test_source_is_set_to_the_path(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        handle.write('{"my-model": {"input": 1, "output": 3}}')
        handle.close()
        try:
            table = load_pricing(handle.name)
            self.assertEqual(table.source, handle.name)
        finally:
            os.remove(handle.name)


if __name__ == "__main__":
    unittest.main()
