import unittest

from llm_cost.usage import UsageRecord, aggregate, load_usage, parse_record


class ParseRecordAnthropicShape(unittest.TestCase):
    def test_basic_fields(self):
        record = parse_record(
            {
                "model": "claude-opus-5",
                "date": "2026-06-01",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                },
            }
        )
        self.assertEqual(record.model, "claude-opus-5")
        self.assertEqual(record.date, "2026-06-01")
        self.assertEqual(record.input_tokens, 100)
        self.assertEqual(record.output_tokens, 50)
        self.assertEqual(record.cached_input_tokens, 20)
        self.assertEqual(record.cache_write_tokens, 5)

    def test_missing_cache_fields_default_to_zero(self):
        record = parse_record(
            {"model": "claude-opus-5", "usage": {"input_tokens": 10, "output_tokens": 4}}
        )
        self.assertEqual(record.cached_input_tokens, 0)
        self.assertEqual(record.cache_write_tokens, 0)


class ParseRecordOpenAIShape(unittest.TestCase):
    def test_prompt_tokens_are_split_into_input_and_cached(self):
        record = parse_record(
            {
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "prompt_tokens_details": {"cached_tokens": 400},
                },
            }
        )
        # OpenAI's prompt_tokens includes the cached prefix; llm_cost normalises
        # to the exclusive form so input_tokens never double-counts a cache read.
        self.assertEqual(record.input_tokens, 600)
        self.assertEqual(record.cached_input_tokens, 400)
        self.assertEqual(record.cache_write_tokens, 0)
        self.assertEqual(record.output_tokens, 200)

    def test_missing_details_default_to_zero_cached(self):
        record = parse_record(
            {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 300, "completion_tokens": 10}}
        )
        self.assertEqual(record.input_tokens, 300)
        self.assertEqual(record.cached_input_tokens, 0)

    def test_cached_exceeding_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_record(
                {
                    "model": "gpt-4o-mini",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "prompt_tokens_details": {"cached_tokens": 150},
                    },
                }
            )


class ParseRecordErrors(unittest.TestCase):
    def test_not_an_object(self):
        with self.assertRaises(ValueError):
            parse_record(["not", "an", "object"])

    def test_missing_model(self):
        with self.assertRaises(ValueError):
            parse_record({"usage": {"input_tokens": 1, "output_tokens": 1}})

    def test_missing_usage(self):
        with self.assertRaises(ValueError):
            parse_record({"model": "claude-opus-5"})

    def test_usage_not_an_object(self):
        with self.assertRaises(ValueError):
            parse_record({"model": "claude-opus-5", "usage": "not-an-object"})

    def test_usage_shape_not_recognised(self):
        with self.assertRaises(ValueError):
            parse_record({"model": "claude-opus-5", "usage": {"weird_field": 1}})

    def test_negative_token_count(self):
        with self.assertRaises(ValueError):
            parse_record(
                {"model": "claude-opus-5", "usage": {"input_tokens": -1, "output_tokens": 1}}
            )

    def test_bool_is_not_a_valid_token_count(self):
        # bool is a subclass of int; parse_record must reject it explicitly
        # rather than silently treating True/False as 1/0 tokens.
        with self.assertRaises(ValueError):
            parse_record(
                {"model": "claude-opus-5", "usage": {"input_tokens": True, "output_tokens": 1}}
            )


class UsageRecordField(unittest.TestCase):
    def test_field_looks_up_model_and_date_directly(self):
        record = UsageRecord(
            model="claude-opus-5",
            date="2026-06-01",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_write_tokens=0,
            raw={"model": "claude-opus-5", "date": "2026-06-01", "team": "agents"},
        )
        self.assertEqual(record.field("model"), "claude-opus-5")
        self.assertEqual(record.field("date"), "2026-06-01")
        self.assertEqual(record.field("team"), "agents")

    def test_field_falls_back_to_unknown_marker(self):
        record = UsageRecord(
            model="claude-opus-5",
            date="",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_write_tokens=0,
            raw={"model": "claude-opus-5"},
        )
        self.assertEqual(record.field("team"), "(unknown)")

    def test_field_date_buckets_a_timestamp_by_calendar_day(self):
        record = UsageRecord(
            model="claude-opus-5",
            date="2026-06-01T09:12:00Z",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_write_tokens=0,
            raw={"model": "claude-opus-5"},
        )
        self.assertEqual(record.field("date"), "2026-06-01")

    def test_field_date_passes_through_a_bare_day(self):
        record = UsageRecord(
            model="claude-opus-5",
            date="2026-06-01",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_write_tokens=0,
            raw={"model": "claude-opus-5"},
        )
        self.assertEqual(record.field("date"), "2026-06-01")

    def test_field_date_falls_back_to_unknown_marker_when_blank(self):
        record = UsageRecord(
            model="claude-opus-5",
            date="",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_write_tokens=0,
            raw={"model": "claude-opus-5"},
        )
        self.assertEqual(record.field("date"), "(unknown)")

    def test_field_date_passes_through_unrecognised_shapes(self):
        record = UsageRecord(
            model="claude-opus-5",
            date="not-a-date",
            input_tokens=1,
            output_tokens=1,
            cached_input_tokens=0,
            cache_write_tokens=0,
            raw={"model": "claude-opus-5"},
        )
        self.assertEqual(record.field("date"), "not-a-date")


class LoadUsage(unittest.TestCase):
    def test_blank_lines_are_skipped(self):
        text = (
            '{"model":"claude-opus-5","usage":{"input_tokens":1,"output_tokens":1}}\n'
            "\n"
            "   \n"
            '{"model":"gpt-4o-mini","usage":{"prompt_tokens":2,"completion_tokens":1}}\n'
        )
        records, problems = load_usage(text)
        self.assertEqual(len(records), 2)
        self.assertEqual(problems, [])

    def test_malformed_line_is_collected_not_raised(self):
        text = '{"model":"claude-opus-5","usage":{"input_tokens":1,"output_tokens":1}}\nnot json\n'
        records, problems = load_usage(text)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(problems), 1)
        self.assertIn("line 2", problems[0])

    def test_strict_raises_on_first_bad_line(self):
        text = "not json\n"
        with self.assertRaises(ValueError):
            load_usage(text, strict=True)


class Aggregate(unittest.TestCase):
    def test_groups_preserve_first_seen_order(self):
        records, _ = load_usage(
            '{"model":"b","usage":{"input_tokens":1,"output_tokens":1}}\n'
            '{"model":"a","usage":{"input_tokens":1,"output_tokens":1}}\n'
            '{"model":"b","usage":{"input_tokens":1,"output_tokens":1}}\n'
        )
        groups = aggregate(records, group_by="model")
        self.assertEqual(list(groups.keys()), ["b", "a"])
        self.assertEqual(len(groups["b"]), 2)
        self.assertEqual(len(groups["a"]), 1)


if __name__ == "__main__":
    unittest.main()
