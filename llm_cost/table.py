"""Number formatting and the fixed-width table renderer used by the CLI."""

__all__ = ["format_int", "format_money", "render_table"]


def format_int(value):
    return "{:,}".format(int(value))


def format_money(value):
    return "$%.4f" % value


def render_table(headers, rows):
    """Render a table of pre-formatted string cells.

    The first column is left-aligned (it holds labels); every other column is
    right-aligned (they hold numbers). Column widths are the max of the
    header and every cell in that column, columns are separated by two
    spaces, and a dashed rule sits under the header.
    """
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(cells):
        padded = []
        for index, cell in enumerate(cells):
            if index == 0:
                padded.append(cell.ljust(widths[index]))
            else:
                padded.append(cell.rjust(widths[index]))
        return "  ".join(padded).rstrip()

    lines = [render_row(headers), "  ".join("-" * width for width in widths)]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)
