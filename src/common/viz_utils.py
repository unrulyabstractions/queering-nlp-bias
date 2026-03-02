"""Text-based visualization utilities for console output."""

from __future__ import annotations

import math


# ══════════════════════════════════════════════════════════════════════════════
# Text Formatting
# ══════════════════════════════════════════════════════════════════════════════


def truncate(text: str, max_len: int = 50, suffix: str = "...") -> str:
    """Truncate text to max_len, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def preview(text: str, max_len: int = 50) -> str:
    """Create a preview of text, truncating with ellipsis if needed."""
    return truncate(text, max_len, "...")


def wrap_text(text: str, width: int = 78, indent: str = "  ") -> list[str]:
    """Wrap text to width with indent prefix on each line."""
    words = text.split()
    if not words:
        return []

    lines = []
    line = indent
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = indent + word
        else:
            line = line + " " + word if line != indent else indent + word
    if line.strip():
        lines.append(line)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Numeric Utilities
# ══════════════════════════════════════════════════════════════════════════════


def sanitize_float(x: float) -> float:
    """Replace inf/nan with finite values."""
    if math.isnan(x):
        return 0.0
    if math.isinf(x):
        return -1000.0 if x < 0 else 1000.0
    return x


def sanitize_floats(values: list[float]) -> list[float]:
    """Sanitize a list of floats."""
    return [sanitize_float(x) for x in values]


# ══════════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════════


def compute_percentiles(values: list[float], percentiles: list[int]) -> dict[int, float]:
    """Compute percentiles of a list of values."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    result = {}
    for p in percentiles:
        idx = int(p / 100 * (n - 1))
        result[p] = sorted_vals[idx]
    return result


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {}

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = variance ** 0.5

    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": std,
        "n": n,
    }


def format_histogram_vertical(
    values: list[float],
    num_bins: int = 10,
    bar_char: str = "█",
    width: int = 30,
) -> list[str]:
    """Create a vertical histogram (bins on Y axis, bars horizontal).

    Returns list of lines to print.
    """
    if not values:
        return ["  No data"]

    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        return [f"  All values = {min_v:.2f}"]

    # Create bins
    bin_width = (max_v - min_v) / num_bins
    bins = [0] * num_bins

    for v in values:
        bin_idx = min(int((v - min_v) / bin_width), num_bins - 1)
        bins[bin_idx] += 1

    # Find max count for scaling
    max_count = max(bins) if bins else 1

    lines = []
    lines.append(f"  {'Range':<12} {'Count':<6} Distribution")
    lines.append("  " + "─" * 50)

    for i, count in enumerate(bins):
        lo = min_v + i * bin_width
        hi = min_v + (i + 1) * bin_width
        bar_len = int((count / max_count) * width) if max_count > 0 else 0
        bar = bar_char * bar_len
        lines.append(f"  {lo:5.2f}-{hi:5.2f}  {count:<6} {bar}")

    return lines


def format_sequence_plot(
    values: list[float],
    max_width: int = 80,
    height: int = 10,
    label: str = "Value",
    show_mean: bool = True,
    show_stats: bool = True,
) -> list[str]:
    """Create a horizontal plot showing values over sequence positions.

    X axis: position in sequence (1:1 mapping when possible, binned only if needed)
    Y axis: value magnitude

    Uses Unicode block characters for smooth gradients:
    ▁▂▃▄▅▆▇█ (8 levels per cell for sub-row precision)

    Args:
        values: List of values to plot
        max_width: Maximum plot width (uses all positions up to this limit)
        height: Plot height in rows
        label: Y-axis label
        show_mean: Show mean line marker
        show_stats: Show min/max position annotations

    Returns list of lines to print.
    """
    if not values:
        return ["  No data"]

    n = len(values)
    min_v, max_v = min(values), max(values)

    # Handle constant values
    if min_v == max_v:
        return [f"  All positions have {label} = {min_v:.2f}"]

    # Block characters for 8 sub-levels (empty, ▁, ▂, ▃, ▄, ▅, ▆, ▇, █)
    blocks = " ▁▂▃▄▅▆▇█"

    # Use 1:1 mapping when possible, only bin if necessary
    num_cols = min(max_width, n)
    bin_size = n / num_cols
    is_binned = bin_size > 1.0

    # Compute binned/mapped values
    binned = []
    bin_positions = []  # Track which original positions map to each column
    for i in range(num_cols):
        start = int(i * bin_size)
        end = int((i + 1) * bin_size)
        end = max(end, start + 1)  # At least one position
        bin_positions.append((start, end))
        if end > start:
            bin_avg = sum(values[start:end]) / (end - start)
            binned.append(bin_avg)
        else:
            binned.append(values[start] if start < n else 0)

    # Find min/max positions
    min_pos = values.index(min_v)
    max_pos = values.index(max_v)
    min_col = int(min_pos / bin_size) if is_binned else min_pos
    max_col = int(max_pos / bin_size) if is_binned else max_pos
    min_col = min(min_col, num_cols - 1)
    max_col = min(max_col, num_cols - 1)

    # Normalize to height (in sub-levels: height * 8)
    range_v = max_v - min_v
    total_levels = height * 8
    levels = [int((v - min_v) / range_v * total_levels) for v in binned]

    # Compute mean level for marker
    mean_v = sum(values) / n
    mean_level = int((mean_v - min_v) / range_v * total_levels)
    mean_row = height - 1 - (mean_level // 8)  # Which row (0 = top)

    # Build rows from top to bottom
    lines = []

    # Y-axis width for labels
    y_width = 6

    for row in range(height):
        row_base = (height - 1 - row) * 8  # Base level for this row

        # Y-axis label (show at top, middle, bottom)
        if row == 0:
            y_label = f"{max_v:>{y_width}.2f}"
        elif row == height - 1:
            y_label = f"{min_v:>{y_width}.2f}"
        elif row == height // 2:
            mid_v = (max_v + min_v) / 2
            y_label = f"{mid_v:>{y_width}.2f}"
        else:
            y_label = " " * y_width

        # Build the plot row
        line = y_label + " │"
        for col, level in enumerate(levels):
            if level >= row_base + 8:
                line += blocks[8]  # Full block
            elif level > row_base:
                line += blocks[level - row_base]  # Partial block
            else:
                line += " "

        # Annotations on right side
        if show_mean and row == mean_row:
            line += f" ← μ={mean_v:.2f}"
        elif show_stats and row == 0:
            line += f" ← max @{max_pos}"
        elif show_stats and row == height - 1:
            line += f" ← min @{min_pos}"

        lines.append(line)

    # X-axis line with tick marks
    x_axis = " " * y_width + " └"
    tick_interval = max(1, num_cols // 5)  # ~5 ticks
    for i in range(num_cols):
        if i % tick_interval == 0:
            x_axis += "┴"
        else:
            x_axis += "─"
    lines.append(x_axis)

    # X-axis tick labels
    tick_line = " " * (y_width + 2)
    last_end = 0
    for i in range(0, num_cols, tick_interval):
        pos = bin_positions[i][0] if is_binned else i
        label_str = str(pos)
        gap = i - last_end
        if gap > 0:
            tick_line += " " * (gap - len(label_str) // 2)
        tick_line += label_str
        last_end = i + len(label_str)
    # Add final position
    final_pos = n - 1
    final_str = str(final_pos)
    remaining = num_cols - last_end
    if remaining > len(final_str):
        tick_line += " " * (remaining - len(final_str)) + final_str
    lines.append(tick_line)

    # Info line
    info = f"n={n}"
    if is_binned:
        info += f" (binned {bin_size:.1f}:1)"
    lines.append(" " * (y_width + 2) + " " * ((num_cols - len(info)) // 2) + info)

    return lines


def print_lines(lines: list[str], log_fn=None) -> None:
    """Print lines using the provided log function or print."""
    printer = log_fn or print
    for line in lines:
        printer(line)
