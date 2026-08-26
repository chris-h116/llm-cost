import unittest

from llm_cost.table import format_int, format_money, render_table


class FormatInt(unittest.TestCase):
    def test_adds_thousands_separators(self):
        self.assertEqual(format_int(1234567), "1,234,567")

    def test_truncates_floats_towards_zero(self):
        self.assertEqual(format_int(1999.9), "1,999")

    def test_small_number_has_no_separator(self):
        self.assertEqual(format_int(42), "42")


class FormatMoney(unittest.TestCase):
    def test_formats_to_four_decimal_places(self):
        self.assertEqual(format_money(1.5), "$1.5000")

    def test_rounds_a_fifth_decimal_place(self):
        self.assertEqual(format_money(0.123456), "$0.1235")

    def test_zero(self):
        self.assertEqual(format_money(0.0), "$0.0000")


class RenderTable(unittest.TestCase):
    def test_first_column_is_left_aligned_others_right_aligned(self):
        rendered = render_table(
            ["model", "cost"], [["claude-opus-5", "$1.0000"], ["gpt-4o", "$12.5000"]]
        )
        lines = rendered.splitlines()
        self.assertTrue(lines[2].startswith("claude-opus-5"))
        self.assertTrue(lines[3].startswith("gpt-4o "))
        self.assertTrue(lines[3].endswith("$12.5000"))
        # gpt-4o's row is shorter than claude-opus-5's, so it is padded to
        # match: the label column stays a fixed width across every row.
        label_width = len("claude-opus-5")
        self.assertEqual(lines[3][:label_width], "gpt-4o".ljust(label_width))

    def test_column_width_is_max_of_header_and_cells(self):
        rendered = render_table(["h"], [["short"], ["a much longer cell"]])
        lines = rendered.splitlines()
        width = len("a much longer cell")
        self.assertEqual(len(lines[0]), width)
        self.assertEqual(len(lines[1]), width)

    def test_rule_line_is_dashes_matching_column_widths(self):
        rendered = render_table(["a", "bb"], [["x", "yy"]])
        rule = rendered.splitlines()[1]
        self.assertEqual(rule, "-  --")

    def test_no_rows_still_renders_header_and_rule(self):
        rendered = render_table(["a", "b"], [])
        self.assertEqual(rendered, "a  b\n-  -")

    def test_trailing_whitespace_is_stripped_from_each_row(self):
        rendered = render_table(["a", "b"], [["x", ""]])
        for line in rendered.splitlines():
            self.assertEqual(line, line.rstrip())


if __name__ == "__main__":
    unittest.main()
