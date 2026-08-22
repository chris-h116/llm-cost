import contextlib
import io
import json
import os
import tempfile
import unittest

from llm_cost import __version__
from llm_cost.cli import main
from llm_cost.pricing import BUILTIN_PRICING, PRICING_AS_OF


def _run(argv):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class VersionFlag(unittest.TestCase):
    def test_version_prints_and_exits_zero(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            with self.assertRaises(SystemExit) as raised:
                main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertIn(__version__, stdout.getvalue())


class EstimateCommand(unittest.TestCase):
    def test_json_output(self):
        code, out, _ = _run(
            ["--json", "estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "500"]
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["model"], "claude-opus-5")
        self.assertEqual(payload["input_tokens"], 1000)
        self.assertEqual(payload["output_tokens"], 500)
        self.assertAlmostEqual(payload["total_cost"], 0.0175)
        self.assertAlmostEqual(payload["cost_per_call"], 0.0175)

    def test_text_output(self):
        code, out, _ = _run(
            ["estimate", "--model", "claude-opus-5", "--input", "1000", "--output", "500"]
        )
        self.assertEqual(code, 0)
        self.assertIn("claude-opus-5", out)
        self.assertIn("cost per call: $0.0175", out)

    def test_unknown_model_exits_3(self):
        code, _, err = _run(["estimate", "--model", "not-a-real-model", "--input", "1", "--output", "1"])
        self.assertEqual(code, 3)
        self.assertIn("no price for model", err)


class ReportCommand(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        handle.write(
            '{"model":"claude-opus-5","team":"agents",'
            '"usage":{"input_tokens":100000,"output_tokens":10000}}\n'
        )
        handle.write(
            '{"model":"gpt-4o-mini","team":"search",'
            '"usage":{"prompt_tokens":50000,"completion_tokens":5000,'
            '"prompt_tokens_details":{"cached_tokens":10000}}}\n'
        )
        handle.write("not json\n")
        handle.write('{"model":"unknown-model-x","usage":{"input_tokens":1,"output_tokens":1}}\n')
        handle.close()
        self.path = handle.name

    def tearDown(self):
        os.remove(self.path)

    def test_default_group_by_model(self):
        code, out, _ = _run(["report", self.path])
        self.assertEqual(code, 0)
        self.assertIn("TOTAL", out)
        # claude-opus-5 costs more than gpt-4o-mini and sorts first.
        self.assertLess(out.index("claude-opus-5"), out.index("gpt-4o-mini"))
        self.assertIn("skipped 1 record(s) with no price: unknown-model-x", out)
        self.assertIn("skipped 1 malformed line(s):", out)

    def test_group_by_other_field(self):
        code, out, _ = _run(["report", self.path, "--group-by", "team"])
        self.assertEqual(code, 0)
        self.assertIn("agents", out)
        self.assertIn("search", out)

    def test_json_output_includes_problems(self):
        code, out, _ = _run(["--json", "report", self.path])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["skipped_records"], 1)
        self.assertEqual(payload["unknown_models"], ["unknown-model-x"])
        self.assertEqual(len(payload["problems"]), 1)

    def test_strict_raises_on_malformed_line(self):
        code, _, err = _run(["report", self.path, "--strict"])
        self.assertEqual(code, 2)
        self.assertIn("line", err)

    def test_missing_file_exits_2(self):
        code, _, err = _run(["report", "/no/such/path/usage.jsonl"])
        self.assertEqual(code, 2)
        self.assertIn("cannot read", err)


class CompareCommand(unittest.TestCase):
    def test_ranked_cheapest_first(self):
        code, out, _ = _run(
            ["compare", "--input", "1000", "--output", "500", "--provider", "anthropic"]
        )
        self.assertEqual(code, 0)
        self.assertLess(out.index("claude-haiku-4-5"), out.index("claude-opus-5"))
        self.assertIn("1.0x", out)

    def test_models_shortlist(self):
        code, out, _ = _run(["compare", "--input", "100", "--output", "100", "--models", "gpt-4o-mini,o3-mini"])
        self.assertEqual(code, 0)
        self.assertIn("gpt-4o-mini", out)
        self.assertIn("o3-mini", out)
        self.assertNotIn("claude-opus-5", out)

    def test_no_matches_is_an_error(self):
        code, _, err = _run(["compare", "--input", "1", "--output", "1", "--provider", "no-such-provider"])
        self.assertEqual(code, 2)
        self.assertIn("no models matched", err)


class ModelsCommand(unittest.TestCase):
    def test_text_output_lists_every_model(self):
        code, out, _ = _run(["models"])
        self.assertEqual(code, 0)
        self.assertIn("%d models" % len(BUILTIN_PRICING), out)

    def test_json_output(self):
        code, out, _ = _run(["--json", "models"])
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["as_of"], PRICING_AS_OF)
        self.assertEqual(payload["source"], "built-in")
        self.assertEqual(len(payload["models"]), len(BUILTIN_PRICING))


class PricingOverride(unittest.TestCase):
    def test_missing_pricing_file_exits_2(self):
        code, _, err = _run(["--pricing", "/no/such/prices.json", "models"])
        self.assertEqual(code, 2)
        self.assertIn("pricing file not found", err)

    def test_override_file_is_applied(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        handle.write(json.dumps({"as_of": "2026-07-01", "models": {"my-model": {"input": 1, "output": 3}}}))
        handle.close()
        try:
            code, out, _ = _run(
                ["--pricing", handle.name, "--json", "estimate", "--model", "my-model", "--input", "1000000", "--output", "0"]
            )
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertAlmostEqual(payload["total_cost"], 1.0)
        finally:
            os.remove(handle.name)


if __name__ == "__main__":
    unittest.main()
