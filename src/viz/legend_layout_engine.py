"""Legend layout engine for tree visualizations.

Pure math algorithms for computing legend layouts:
- Text width estimation and wrapping
- First Fit Decreasing bin packing for row layout
- Constraint-aware placement optimization

No matplotlib code - this module is renderer-agnostic.
"""

from __future__ import annotations

import warnings
from typing import Any

from .viz_bounding_box import (
    BoundingBox,
    TreeContentTracker,
    compute_collision_score,
    compute_legend_bounds,
)

###############################################################################
# TEXT WIDTH ESTIMATION
###############################################################################

# Constants for character-aware width estimation
_WIDE_CHARS = frozenset("MWmw@%#&QDOGHUB")
_NARROW_CHARS = frozenset("il1!|,.:;'`()[]{}fjrt ")


def estimate_text_width(text: str, char_width: float) -> float:
    """Estimate text width with character-aware calculation.

    Wide chars (M, W, etc): 1.4x base width
    Narrow chars (i, l, etc): 0.5x base width
    Others: 1.0x base width
    """
    width = 0.0
    for char in text:
        if char in _WIDE_CHARS:
            width += char_width * 1.4
        elif char in _NARROW_CHARS:
            width += char_width * 0.5
        else:
            width += char_width
    return width


def truncate_to_width(text: str, max_width: float, char_width: float) -> str:
    """Truncate text with ellipsis to fit within max_width."""
    if estimate_text_width(text, char_width) <= max_width:
        return text

    ellipsis = "..."
    ellipsis_w = estimate_text_width(ellipsis, char_width)
    target = max_width - ellipsis_w

    for i in range(len(text), 0, -1):
        candidate = text[:i].rstrip()
        if estimate_text_width(candidate, char_width) <= target:
            return candidate + ellipsis
    return ellipsis


def wrap_to_lines(text: str, max_width: float, char_width: float) -> list[str]:
    """Wrap text to multiple lines to fit within max_width.

    Returns list of lines (no truncation, full text preserved).
    """
    if estimate_text_width(text, char_width) <= max_width:
        return [text]

    words = text.split()
    lines = []
    current_line: list[str] = []
    current_width = 0.0
    space_width = estimate_text_width(" ", char_width)

    for word in words:
        word_width = estimate_text_width(word, char_width)

        # Check if adding this word exceeds max_width
        if current_line:
            test_width = current_width + space_width + word_width
        else:
            test_width = word_width

        if test_width <= max_width:
            current_line.append(word)
            current_width = test_width
        else:
            # Start new line
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width

    # Don't forget the last line
    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [text]


###############################################################################
# LEGEND LAYOUT CONSTANTS
###############################################################################

# Fixed parameters for legend layout
SWATCH_SIZE = 0.22  # Size of color swatch square (large for visibility)
SWATCH_GAP = 0.07  # Gap between swatch and text
ITEM_GAP = 0.22  # Horizontal gap between items (tighter for more coverage)
MARGIN = 0.06  # Left/right margin (minimal for more text space)
MIN_CHAR_WIDTH = 0.09  # Minimum readable char width (~15pt minimum)
MAX_CHAR_WIDTH = 0.22  # Maximum char width (~36pt, fills available space)
MAX_GAP_UNITS = 4.0  # Maximum gap between legend and tree content (allows centering)
LEGEND_TREE_GAP = 0.3  # Minimum gap above root/trunk node
EPSILON_DISTANCE = 0.08  # Minimum clearance between legend and content (with tolerance)
MIN_COVERAGE_RATIO = (
    0.20  # Legend must cover at least 20% of available white space (allows centering)
)


###############################################################################
# TARGET REGION DEFINITION
###############################################################################


def define_target_region(
    figure_bounds: BoundingBox,
    tree_content: TreeContentTracker,
    target_quadrant: str = "top_left",
) -> BoundingBox:
    """Define target region for legend placement.

    CRITICAL CONSTRAINT: Legend must ALWAYS be BELOW the title.
    The target region is bounded by:
    - TOP: title bottom (if title exists) or figure top
    - BOTTOM: global tree content top (including all branches)

    Args:
        figure_bounds: Overall figure bounds
        tree_content: Tracker with tree element positions
        target_quadrant: "top_left", "top_right", "top_center"

    Returns:
        BoundingBox defining ideal legend region
    """
    tree_bounds = tree_content.get_tree_bounds()
    title_box = tree_content.title_box

    # Determine horizontal bounds based on quadrant and actual tree content

    if target_quadrant == "top_left":
        region_x_min = figure_bounds.x_min
        # Use HALF the figure width for legend (tree will shift right to accommodate)
        # This allows larger fonts and better use of white space
        region_x_max = figure_bounds.x_min + figure_bounds.width / 2
    elif target_quadrant == "top_right":
        region_x_min = figure_bounds.width / 2
        region_x_max = figure_bounds.x_max
    else:  # top_center
        region_x_min = figure_bounds.x_min
        region_x_max = figure_bounds.x_max

    # TOP: If title exists, legend must be BELOW it
    if title_box is not None:
        region_top = title_box.y_min
    else:
        region_top = figure_bounds.y_max

    # BOTTOM: Above GLOBAL tree content (including all branches)
    content_bounds = tree_content.get_content_bounds()
    if content_bounds is not None:
        region_bottom = content_bounds.y_max
    elif tree_bounds is not None:
        region_bottom = tree_bounds.y_max
    else:
        region_bottom = figure_bounds.y_max - figure_bounds.height / 4

    # Ensure valid region (top > bottom)
    if region_top <= region_bottom:
        region_top = region_bottom + 0.5

    return BoundingBox(
        region_x_min,
        region_bottom,
        region_x_max,
        region_top,
    )


###############################################################################
# LEGEND LAYOUT ALGORITHM
###############################################################################


def compute_legend_layout(
    descriptions: list[str],
    available_width: float,
    *,
    char_width: float = 0.055,
    swatch_size: float = SWATCH_SIZE,
    swatch_gap: float = SWATCH_GAP,
    item_gap: float = ITEM_GAP,
    row_height: float = 0.32,
    margin: float = MARGIN,
    max_item_fraction: float = 0.40,
    force_n_cols: int | None = None,
) -> dict[str, Any]:
    """Compute legend layout using GRID-BASED column layout.

    Algorithm:
    1. Determine optimal number of columns based on available width
    2. Calculate fixed column width (all columns same width)
    3. Place items in columns round-robin (semantic order preserved)
    4. All items in same column get same x position (proper alignment)

    Args:
        descriptions: List of text descriptions for legend items
        available_width: Total width to fit legend into
        char_width: Width per average character in data units
        swatch_size: Size of color swatch square
        swatch_gap: Gap between swatch and text
        item_gap: Gap between items in same row
        row_height: Height of each row
        margin: Margin on left and right
        max_item_fraction: Max item width as fraction of effective width
        force_n_cols: If set, force this many columns instead of auto-detecting

    Returns:
        dict with:
            items: list of {index, description, swatch_x, swatch_y, text_x, text_y}
            n_rows: number of rows
            total_height: total legend height
            total_width: available_width (unchanged)
            swatch_size: for renderer
    """
    if not descriptions:
        return {
            "items": [],
            "n_rows": 0,
            "total_height": 0.0,
            "total_width": available_width,
            "swatch_size": swatch_size,
        }

    n_items = len(descriptions)
    effective_width = available_width - 2 * margin
    LEFT_NUDGE = 0.15  # Nudge legend right for visual balance

    # Step 1: Determine number of columns
    if force_n_cols is not None:
        # Use forced column count
        n_cols = min(force_n_cols, n_items)
        col_width = (effective_width - item_gap * (n_cols - 1)) / n_cols
        text_width = col_width - swatch_size - swatch_gap
        max_desc_len = max(len(d) for d in descriptions)
        char_width = text_width / max(max_desc_len / 2, 10)
        char_width = max(MIN_CHAR_WIDTH, min(MAX_CHAR_WIDTH, char_width))
    else:
        # Auto-detect: Try 1, 2, 3 columns and pick the one with best readability
        best_n_cols = 1
        best_char_width = char_width

        for test_n_cols in range(1, min(n_items + 1, 4)):  # Max 3 columns
            # Column width with equal spacing
            col_width = (effective_width - item_gap * (test_n_cols - 1)) / test_n_cols
            text_width = col_width - swatch_size - swatch_gap

            if text_width < 0.5:
                continue  # Column too narrow

            # Calculate char_width that fits descriptions
            max_desc_len = max(len(d) for d in descriptions)
            needed_char_width = text_width / max(
                max_desc_len / 2, 10
            )  # Allow 2-line wrap
            needed_char_width = max(
                MIN_CHAR_WIDTH, min(MAX_CHAR_WIDTH, needed_char_width)
            )

            # Pick layout with largest char_width (most readable)
            if needed_char_width >= best_char_width:
                best_char_width = needed_char_width
                best_n_cols = test_n_cols

        n_cols = best_n_cols
        char_width = best_char_width

    n_rows = (n_items + n_cols - 1) // n_cols  # Ceiling division

    # Step 2: Calculate MINIMUM column width needed at chosen char_width
    # First, wrap all descriptions to find actual text widths
    line_height = char_width * 2.0  # Tight line spacing

    # Calculate max text width that fits within available space
    max_available_text_width = (effective_width - item_gap * (n_cols - 1)) / n_cols - swatch_size - swatch_gap

    # Wrap all descriptions and find actual max line width
    items = []
    max_actual_text_width = 0.0

    for i, desc in enumerate(descriptions):
        desc = " ".join(desc.split())  # Normalize whitespace
        lines = wrap_to_lines(desc, max_available_text_width, char_width)
        n_lines = len(lines)
        item_h = max(swatch_size, n_lines * line_height)

        # Track actual width of wrapped lines
        for line in lines:
            line_width = estimate_text_width(line, char_width)
            max_actual_text_width = max(max_actual_text_width, line_width)

        items.append(
            {
                "idx": i,
                "desc": desc,
                "lines": lines,
                "n_lines": n_lines,
                "h": item_h,
            }
        )

    # Calculate column width based on ACTUAL text needs (not available space)
    # This ensures the legend is only as wide as needed
    col_width = max_actual_text_width + swatch_size + swatch_gap

    # Step 4: Arrange items in grid (row-major order for semantic consistency)
    # Row 0: items 0, 1, 2 (if 3 cols)
    # Row 1: items 3, 4, 5
    # etc.
    grid: list[list] = [[] for _ in range(n_rows)]
    for i, item in enumerate(items):
        row_idx = i // n_cols
        grid[row_idx].append(item)

    # Calculate row heights
    row_heights_actual = []
    for row in grid:
        max_h = max(item["h"] for item in row) if row else row_height
        row_heights_actual.append(max_h)

    total_height = sum(row_heights_actual)

    # Step 5: Compute positions with FIXED column x positions
    result_items = []
    current_y = 0.0

    # Calculate actual content width (all columns + gaps between them)
    actual_content_width = n_cols * col_width + (n_cols - 1) * item_gap

    # LEFT-ALIGN: Position content starting from left margin
    # Tree will be positioned at intrinsic_width + gap, keeping legend close to tree
    col_x_positions = []
    for col_idx in range(n_cols):
        col_x = margin + col_idx * (col_width + item_gap)
        col_x_positions.append(col_x)

    for row_i, row in enumerate(grid):
        row_h = row_heights_actual[row_i]
        y = -(current_y + row_h / 2)

        for col_i, item in enumerate(row):
            x = col_x_positions[col_i]
            result_items.append(
                {
                    "index": item["idx"],
                    "description": item["desc"],
                    "lines": item["lines"],
                    "n_lines": item["n_lines"],
                    "swatch_x": x,
                    "swatch_y": y,
                    "text_x": x + swatch_size + swatch_gap,
                    "text_y": y,
                    "item_height": item["h"],
                }
            )

        current_y += row_h

    # Calculate intrinsic content width (independent of positioning)
    # This is the actual space needed: all columns + gaps + margins
    intrinsic_width = actual_content_width + 2 * margin

    return {
        "items": result_items,
        "n_rows": n_rows,
        "total_height": total_height,
        "total_width": intrinsic_width,  # Actual content width for tree offset
        "swatch_size": swatch_size,
        "char_width": char_width,
        "row_height": row_height,
    }


###############################################################################
# LEGEND PLACEMENT OPTIMIZATION
###############################################################################


def optimize_legend_placement(
    descriptions: list[str],
    tree_content: TreeContentTracker,
    figure_bounds: BoundingBox,
    *,
    target_quadrant: str = "top_left",
    coverage_weight: float = 1.0,
    collision_weight: float = 10.0,
    compactness_weight: float = 0.5,
    debug: bool = False,
) -> tuple[dict[str, Any], float, BoundingBox]:
    """Compute optimal legend placement using rectangle-first approach.

    Algorithm:
    1. Define the target rectangle (width x height)
    2. Try different column counts (1, 2, 3, ...)
    3. For each column count, calculate the resulting text size
    4. Pick the column count that maximizes text readability

    Key constraints:
    - Swatch size is FIXED (constant across all layouts)
    - Legend fills the target rectangle
    - Text size adapts to available space
    - Legend must be below title
    - Legend must not collide with tree content

    Args:
        descriptions: Legend item descriptions
        tree_content: Tracker with all tree element bounds
        figure_bounds: Overall figure bounding box
        target_quadrant: Where to place legend ("top_left", "top_right", "top_center")
        coverage_weight: Unused (kept for API compatibility)
        collision_weight: Unused (kept for API compatibility)
        compactness_weight: Unused (kept for API compatibility)
        debug: If True, print debug information

    Returns:
        Tuple of (best_layout, legend_top_y, target_region)
    """
    if not descriptions:
        return (
            {
                "items": [],
                "n_rows": 0,
                "total_height": 0,
                "total_width": 0,
                "swatch_size": SWATCH_SIZE,
            },
            figure_bounds.y_max,
            figure_bounds,
        )

    # STEP 1: Define the target rectangle
    target_region = define_target_region(figure_bounds, tree_content, target_quadrant)

    # CONSTRAINT: Legend width cannot exceed HALF the figure width
    max_legend_width = figure_bounds.width / 2

    # CONSTRAINT: Legend must not overlap with tree content at the same y level
    all_boxes = tree_content.get_all_content_boxes()
    upper_content = [
        box
        for box in all_boxes
        if box.y_max > target_region.y_min - 1.0  # Content in or above legend zone
    ]

    # If there's upper content that could collide, compute max safe width
    if upper_content:
        right_content = [box for box in upper_content if box.x_min > 0.5]
        if right_content:
            min_right_x = min(box.x_min for box in right_content)
            HORIZONTAL_GAP = 0.3
            max_width_before_tree = min_right_x - figure_bounds.x_min - HORIZONTAL_GAP
            if max_width_before_tree > 0.5:
                max_legend_width = min(max_legend_width, max_width_before_tree)

    rect_width = min(target_region.width, max_legend_width)
    n_items = len(descriptions)

    # STEP 2: Calculate available vertical space
    # Find local tree content on left side (where legend goes)
    left_tree_boxes = [
        box
        for box in tree_content.node_boxes
        if box.x_min < 3.0  # Left third of typical figure
    ]
    if left_tree_boxes:
        local_tree_top = max(box.y_max for box in left_tree_boxes)
    else:
        tree_bounds = tree_content.get_tree_bounds()
        local_tree_top = tree_bounds.y_max if tree_bounds else target_region.y_min

    # Title bottom or figure top
    if tree_content.title_box is not None:
        region_top = tree_content.title_box.y_min - EPSILON_DISTANCE
    else:
        region_top = target_region.y_max

    available_height = region_top - local_tree_top - LEGEND_TREE_GAP
    available_height = max(0.5, available_height)  # At least 0.5 units

    # STEP 3: Try different column counts to find optimal layout
    best_layout = None
    best_score = -1.0  # Score = char_width, but only consider fitting layouts
    effective_width = rect_width - 2 * MARGIN

    if debug:
        print("\n=== LEGEND LAYOUT DEBUG ===")
        print(f"n_items: {len(descriptions)}")
        print(f"descriptions: {descriptions}")
        print(f"figure_bounds: {figure_bounds}")
        print(f"target_region: {target_region}")
        print(f"max_legend_width: {max_legend_width:.2f}")
        print(f"rect_width: {rect_width:.2f}")
        print(f"effective_width: {effective_width:.2f}")
        print(f"region_top: {region_top:.2f}")
        print(f"local_tree_top: {local_tree_top:.2f}")
        print(f"available_height: {available_height:.2f}")
        print(f"MIN_CHAR_WIDTH: {MIN_CHAR_WIDTH}, MAX_CHAR_WIDTH: {MAX_CHAR_WIDTH}")
        print()

    for n_cols in range(1, min(n_items + 1, 5)):  # Try 1-4 columns
        n_rows = (n_items + n_cols - 1) // n_cols  # Ceiling division

        # Available width per column (accounting for gaps)
        col_width = (effective_width - ITEM_GAP * (n_cols - 1)) / n_cols

        # Available text width = column width - swatch - swatch_gap
        text_width_available = col_width - SWATCH_SIZE - SWATCH_GAP

        if text_width_available <= 0:
            continue  # Column too narrow

        # Calculate char_width to fit the average description
        avg_desc_len = sum(len(d) for d in descriptions) / max(n_items, 1)
        max_desc_len = max(len(d) for d in descriptions)

        # Start with char_width that fits average description in one line
        char_width = text_width_available / max(avg_desc_len, 10)

        # Clamp char_width to readable range
        char_width = max(MIN_CHAR_WIDTH, min(MAX_CHAR_WIDTH, char_width))

        # Estimate average lines per item based on max description
        chars_per_line = text_width_available / char_width
        avg_lines_per_item = max(1.0, max_desc_len / max(chars_per_line, 1))
        avg_lines_per_item = min(avg_lines_per_item, 3.0)  # Cap at 3 lines

        # Row height based on content
        line_height = char_width * 2.2  # Comfortable line spacing
        min_row_height = SWATCH_SIZE + 0.06
        max_row_height = 0.45  # Allow taller rows for readability

        row_height = min(
            max_row_height, max(min_row_height, avg_lines_per_item * line_height)
        )

        # Estimate total height
        estimated_height = n_rows * row_height

        # CRITICAL: Skip layouts that don't fit in available height
        if estimated_height > available_height:
            # Try reducing row height to fit
            row_height = available_height / n_rows
            if row_height < min_row_height:
                continue  # Can't fit even with minimum row height
            # Recalculate char_width based on reduced row height
            char_width = row_height / 2.2
            char_width = max(MIN_CHAR_WIDTH, min(MAX_CHAR_WIDTH, char_width))

        # Score: prefer larger char_width (more readable)
        score = char_width

        if debug:
            print(
                f"  n_cols={n_cols}: n_rows={n_rows}, col_width={col_width:.3f}, "
                f"text_width={text_width_available:.3f}, char_width={char_width:.3f}, "
                f"row_height={row_height:.3f}, est_height={estimated_height:.3f}, "
                f"fits={estimated_height <= available_height}, score={score:.3f}"
            )

        if score > best_score:
            best_score = score
            best_layout = {
                "n_cols": n_cols,
                "n_rows": n_rows,
                "row_height": row_height,
                "col_width": col_width,
                "char_width": char_width,
            }

    # Fallback if no valid layout found
    if best_layout is None:
        fallback_row_height = min(0.5, available_height / max(n_items, 1))
        best_layout = {
            "n_cols": 1,
            "n_rows": n_items,
            "row_height": fallback_row_height,
            "col_width": effective_width,
            "char_width": MIN_CHAR_WIDTH,
        }
        if debug:
            print(f"  FALLBACK: n_cols=1, char_width={MIN_CHAR_WIDTH}")

    if debug:
        print(
            f"\n  BEST LAYOUT: n_cols={best_layout['n_cols']}, "
            f"char_width={best_layout['char_width']:.3f}, "
            f"row_height={best_layout['row_height']:.3f}"
        )

    # STEP 4: Compute actual layout and iterate until no collision
    # Start with best layout, fall back to more columns if needed
    layout = None
    legend_top_y = None

    for attempt in range(4):  # Try up to 4 times with more columns
        test_n_cols = best_layout["n_cols"] + attempt
        if test_n_cols > n_items:
            break

        # Recalculate layout parameters for this column count
        n_rows = (n_items + test_n_cols - 1) // test_n_cols
        col_width = (effective_width - ITEM_GAP * (test_n_cols - 1)) / test_n_cols
        text_width = col_width - SWATCH_SIZE - SWATCH_GAP

        if text_width < 0.3:
            continue

        # Calculate char_width for this layout
        max_desc_len = max(len(d) for d in descriptions)
        char_width = text_width / max(max_desc_len / 2, 10)
        char_width = max(MIN_CHAR_WIDTH, min(MAX_CHAR_WIDTH, char_width))

        layout = compute_legend_layout(
            descriptions,
            rect_width,
            char_width=char_width,
            swatch_size=SWATCH_SIZE,
            swatch_gap=SWATCH_GAP,
            item_gap=ITEM_GAP,
            row_height=best_layout["row_height"],
            margin=MARGIN,
            max_item_fraction=0.95,
            force_n_cols=test_n_cols,  # Force specific column count
        )

        # Position legend
        legend_top_y = _compute_legend_position(layout, tree_content, target_region)

        # Check collision
        legend_bounds = compute_legend_bounds(layout, legend_top_y)
        collision = compute_collision_score(
            legend_bounds, tree_content, collision_penalty=1.0
        )

        if debug:
            print(
                f"\n  ATTEMPT {attempt}: test_n_cols={test_n_cols}, "
                f"collision={collision:.3f}, "
                f"layout_char_width={layout.get('char_width', 'N/A')}"
            )

        if collision == 0:
            break  # Found a non-colliding layout

        # Try relayout with tighter width
        layout, legend_top_y = _relayout_to_avoid_collision(
            layout, legend_top_y, tree_content, target_region
        )

        # Check again
        legend_bounds = compute_legend_bounds(layout, legend_top_y)
        collision = compute_collision_score(
            legend_bounds, tree_content, collision_penalty=1.0
        )

        if debug:
            print(
                f"    After relayout: collision={collision:.3f}, "
                f"char_width={layout.get('char_width', 'N/A')}"
            )

        if collision == 0:
            break

    # Ensure legend meets minimum coverage requirement before validation.
    # Save the pre-coverage layout so we can revert if scaling introduces a collision.
    pre_coverage_layout = layout
    pre_coverage_legend_top_y = legend_top_y

    layout, legend_top_y = _ensure_minimum_coverage(
        layout, legend_top_y, target_region, tree_content
    )

    # If coverage scaling introduced a new collision, revert to the unscaled layout.
    scaled_bounds = compute_legend_bounds(layout, legend_top_y)
    if compute_collision_score(scaled_bounds, tree_content, collision_penalty=1.0) > 0:
        layout = pre_coverage_layout
        legend_top_y = pre_coverage_legend_top_y

    # CONSTRAINT VALIDATION
    validate_legend_constraints(layout, legend_top_y, target_region, tree_content)

    if debug:
        print("\n=== FINAL LAYOUT ===")
        print(f"char_width: {layout.get('char_width', 'N/A')}")
        print(f"swatch_size: {layout.get('swatch_size', 'N/A')}")
        print(f"n_rows: {layout.get('n_rows', 'N/A')}")
        print(f"total_width: {layout.get('total_width', 'N/A')}")
        print(f"total_height: {layout.get('total_height', 'N/A')}")
        print(f"legend_top_y: {legend_top_y:.3f}")
        print(f"row_height: {layout.get('row_height', 'N/A')}")
        # Estimate font size: char_width roughly maps to points
        # Standard approximation: char_width of 0.1 is about 16-18pt
        char_w = layout.get("char_width", 0.09)
        est_fontsize = char_w * 160  # rough estimate
        print(f"Estimated font size: ~{est_fontsize:.1f}pt")
        print()

    return layout, legend_top_y, target_region


def _compute_legend_position(
    layout: dict[str, Any],
    tree_content: TreeContentTracker,
    target_region: BoundingBox,
) -> float:
    """Compute legend Y position CENTERED in the available quadrant.

    Centers the legend vertically between the tree content and the title/figure top,
    while respecting the maximum gap constraint.

    Returns legend_top_y coordinate.
    """
    items = layout.get("items", [])
    if items:
        min_item_y = min(item["swatch_y"] for item in items)
        swatch_size = layout.get("swatch_size", SWATCH_SIZE)
        visual_extent_below_top = abs(min_item_y) + swatch_size / 2
    else:
        visual_extent_below_top = layout.get("total_height", 0)

    legend_height = visual_extent_below_top

    # Find LOCAL tree content on LEFT side (where legend goes)
    left_tree_boxes = [
        box
        for box in tree_content.node_boxes
        if box.x_min < 3.0  # Left third of typical figure
    ]
    if left_tree_boxes:
        local_tree_top = max(box.y_max for box in left_tree_boxes)
    else:
        tree_bounds = tree_content.get_tree_bounds()
        local_tree_top = tree_bounds.y_max if tree_bounds else target_region.y_min

    # Determine the top boundary (title bottom or figure top)
    if tree_content.title_box is not None:
        region_top = tree_content.title_box.y_min - EPSILON_DISTANCE
    else:
        region_top = target_region.y_max

    # Calculate available vertical space for centering
    # Available region: from (local_tree_top + min_gap) to (region_top - margin)
    min_legend_bottom = local_tree_top + LEGEND_TREE_GAP  # Minimum position
    max_legend_top = region_top - 0.1  # Maximum position (below title)

    # Available space for the legend
    available_height = max_legend_top - min_legend_bottom

    if available_height > legend_height:
        # CENTER the legend vertically in the available space
        # Extra space above and below the legend
        extra_space = available_height - legend_height
        # Put half the extra space below the legend (raising it up)
        centered_legend_bottom = min_legend_bottom + extra_space / 2
        legend_top_y = centered_legend_bottom + legend_height
    else:
        # Not enough space - place as high as possible
        legend_top_y = max_legend_top

    # Ensure we don't exceed the max gap constraint
    legend_visual_bottom = legend_top_y - legend_height
    gap = legend_visual_bottom - local_tree_top
    if gap > MAX_GAP_UNITS:
        # Pull legend down to respect max gap
        legend_top_y = local_tree_top + MAX_GAP_UNITS + legend_height

    # Final clamp: must be below title
    if tree_content.title_box is not None:
        title_bottom = tree_content.title_box.y_min
        max_legend_top = title_bottom - 0.1
        if legend_top_y > max_legend_top:
            legend_top_y = max_legend_top

    return legend_top_y


def _relayout_to_avoid_collision(
    layout: dict[str, Any],
    legend_top_y: float,
    tree_content: TreeContentTracker,
    target_region: BoundingBox,
) -> tuple[dict[str, Any], float]:
    """Relayout legend with tighter width to avoid collision.

    Returns (new_layout, new_legend_top_y).
    """
    legend_bounds = compute_legend_bounds(layout, legend_top_y)
    legend_y_min = min(b.y_min for b in legend_bounds)
    legend_y_max = max(b.y_max for b in legend_bounds)

    # Find tree content that overlaps in y
    safe_width = target_region.width
    for box in tree_content.node_boxes + tree_content.node_labels:
        if box.y_max > legend_y_min and box.y_min < legend_y_max:
            # This box could collide - legend must end before box.x_min
            if box.x_min > 0.5:
                safe_width = min(safe_width, box.x_min - 0.1 - 0.3)

    if safe_width <= 1.0:
        # Not enough space to relayout
        return layout, legend_top_y

    # Relayout with tighter width
    new_layout = compute_legend_layout(
        descriptions=[item["description"] for item in layout["items"]],
        available_width=safe_width,
        swatch_size=layout["swatch_size"],
        row_height=layout.get("row_height", 0.32),
        margin=MARGIN,
        max_item_fraction=0.95,
    )

    # Recalculate position
    new_legend_top_y = _compute_legend_position(new_layout, tree_content, target_region)

    return new_layout, new_legend_top_y


def _ensure_minimum_coverage(
    layout: dict[str, Any],
    legend_top_y: float,
    target_region: BoundingBox,
    tree_content: TreeContentTracker,
) -> tuple[dict[str, Any], float]:
    """Scale up legend if needed to meet minimum coverage requirement.

    When there are few legend items, the legend may be too small relative
    to the available white space. This function scales up the legend
    dimensions to ensure it fills at least MIN_COVERAGE_RATIO of the
    available region.

    Returns (updated_layout, updated_legend_top_y).
    """
    items = layout.get("items", [])
    if not items:
        return layout, legend_top_y

    # Compute available region (same as in validate_legend_constraints)
    left_tree_boxes = [
        box for box in tree_content.node_boxes if box.x_min < 3.0
    ]
    if left_tree_boxes:
        local_tree_top = max(box.y_max for box in left_tree_boxes)
    else:
        tree_bounds = tree_content.get_tree_bounds()
        local_tree_top = tree_bounds.y_max if tree_bounds else target_region.y_min

    available_region = _compute_available_region(
        target_region, local_tree_top, legend_top_y, tree_content
    )

    if available_region.area <= 0.5:
        return layout, legend_top_y

    # Compute current legend area
    legend_bounds = compute_legend_bounds(layout, legend_top_y)
    legend_union = BoundingBox.union(legend_bounds)
    if not legend_union:
        return layout, legend_top_y

    current_coverage = legend_union.area / available_region.area

    if current_coverage >= MIN_COVERAGE_RATIO:
        return layout, legend_top_y

    # Need to scale up. Compute required scale factor.
    required_area = MIN_COVERAGE_RATIO * available_region.area
    # Scale both width and height proportionally
    scale_factor = (required_area / legend_union.area) ** 0.5

    # Scale up layout dimensions
    new_layout = layout.copy()
    new_layout["items"] = [item.copy() for item in layout["items"]]

    # Scale swatch size
    old_swatch = layout.get("swatch_size", SWATCH_SIZE)
    new_swatch = old_swatch * scale_factor
    new_layout["swatch_size"] = new_swatch

    # Scale row height
    old_row_height = layout.get("row_height", 0.32)
    new_row_height = old_row_height * scale_factor
    new_layout["row_height"] = new_row_height

    # Scale char width (for text rendering)
    old_char_width = layout.get("char_width", 0.09)
    new_char_width = old_char_width * scale_factor
    new_layout["char_width"] = new_char_width

    # Scale total dimensions
    new_layout["total_width"] = layout.get("total_width", 1.0) * scale_factor
    new_layout["total_height"] = layout.get("total_height", 0.5) * scale_factor

    # Scale item positions
    swatch_scale = new_swatch / old_swatch
    for item in new_layout["items"]:
        item["swatch_x"] = item["swatch_x"] * scale_factor
        item["swatch_y"] = item["swatch_y"] * scale_factor
        item["text_x"] = item["text_x"] * scale_factor
        item["text_y"] = item["text_y"] * scale_factor
        item["swatch_size"] = new_swatch

    return new_layout, legend_top_y


###############################################################################
# CONSTRAINT VALIDATION
###############################################################################


def validate_legend_constraints(
    layout: dict[str, Any],
    legend_top_y: float,
    target_region: BoundingBox,
    tree_content: TreeContentTracker,
) -> None:
    """Validate legend placement constraints and ASSERT if violated.

    Constraints:
    1. Gap to LOCAL tree content <= MAX_GAP_UNITS
    2. Legend must be BELOW title
    3. No collision with tree content
    4. Epsilon distance maintained from all content (no near-touches)
    5. Minimum coverage of available white space in upper-left quadrant
    """
    items = layout.get("items", [])
    if not items:
        return

    # Calculate legend visual bottom
    min_item_y = min(item["swatch_y"] for item in items)
    swatch_size = layout.get("swatch_size", SWATCH_SIZE)
    legend_visual_bottom = legend_top_y + min_item_y - swatch_size / 2

    # Get tree bounds
    tree_bounds = tree_content.get_tree_bounds()
    tree_node_top = tree_bounds.y_max if tree_bounds else target_region.y_min

    # CONSTRAINT 1: Gap between legend bottom and LOCAL tree content on LEFT side
    left_tree_boxes = [
        box
        for box in tree_content.node_boxes
        if box.x_min < 3.0  # Left third of typical figure
    ]
    if left_tree_boxes:
        local_tree_top = max(box.y_max for box in left_tree_boxes)
    else:
        local_tree_top = tree_node_top

    gap = legend_visual_bottom - local_tree_top
    # Use small tolerance for floating point comparison
    TOLERANCE = 0.01
    assert gap <= MAX_GAP_UNITS + TOLERANCE, (
        f"Legend gap constraint FAILED: gap={gap:.2f} > max={MAX_GAP_UNITS} "
        f"(legend_bottom={legend_visual_bottom:.2f}, local_tree_top={local_tree_top:.2f})"
    )

    # CONSTRAINT 2: Legend must be BELOW title (if title exists)
    if tree_content.title_box is not None:
        title_bottom = tree_content.title_box.y_min
        assert legend_top_y < title_bottom, (
            f"Legend must be below title: legend_top={legend_top_y:.2f} >= title_bottom={title_bottom:.2f}"
        )

    # CONSTRAINT 3: Legend must not overlap tree content (nodes, labels, title)
    legend_bounds = compute_legend_bounds(layout, legend_top_y)
    collision = compute_collision_score(
        legend_bounds, tree_content, collision_penalty=1.0
    )
    if collision > 0:
        warnings.warn(
            f"Legend collision could not be fully resolved: score={collision:.2f}. "
            "This can happen when the tree has no branch arms.",
            stacklevel=4,
        )
        return

    # CONSTRAINT 4: Epsilon distance from all tree content
    content_boxes = list(tree_content.node_boxes) + list(tree_content.node_labels)
    if tree_content.title_box:
        content_boxes.append(tree_content.title_box)

    for legend_box in legend_bounds:
        for content_box in content_boxes:
            # Skip content that's clearly far away (more than 2 units apart in y)
            if abs(legend_box.center_y - content_box.center_y) > 2.0:
                continue

            distance = legend_box.distance_to(content_box)
            # Epsilon check only applies to non-overlapping boxes
            if 0 < distance < EPSILON_DISTANCE:
                assert False, (
                    f"Legend epsilon distance FAILED: distance={distance:.3f} < "
                    f"epsilon={EPSILON_DISTANCE} between legend and content"
                )

    # CONSTRAINT 5: Minimum coverage of available white space
    # Calculate the available region (upper-left quadrant above local tree)
    available_region = _compute_available_region(
        target_region, local_tree_top, legend_top_y, tree_content
    )

    if available_region.area > 0:
        # Calculate legend bounding box (union of all items)
        legend_union = BoundingBox.union(legend_bounds)
        if legend_union:
            coverage = legend_union.area / available_region.area
            # Only enforce for regions with meaningful space
            # Use small tolerance for floating point comparison
            COVERAGE_TOLERANCE = 0.001
            if available_region.area > 0.5:
                assert coverage >= MIN_COVERAGE_RATIO - COVERAGE_TOLERANCE, (
                    f"Legend coverage constraint FAILED: coverage={coverage:.1%} < "
                    f"min={MIN_COVERAGE_RATIO:.0%} (legend_area={legend_union.area:.2f}, "
                    f"available_area={available_region.area:.2f})"
                )


def _compute_available_region(
    target_region: BoundingBox,
    local_tree_top: float,
    legend_top_y: float,
    tree_content: TreeContentTracker,
) -> BoundingBox:
    """Compute the available white space region for legend placement.

    This is the region where the legend can be placed:
    - Horizontally: from figure left to the nearest tree content
    - Vertically: from local tree top to legend top (or title bottom)
    """
    # Horizontal bounds: from left edge to nearest tree content on the right
    x_min = target_region.x_min
    x_max = target_region.x_max

    # Find tree content that could constrain legend width
    for box in tree_content.node_boxes:
        if box.y_max > local_tree_top - 0.5:  # Content near legend level
            if box.x_min > 0.5:  # Content to the right
                x_max = min(x_max, box.x_min - EPSILON_DISTANCE)

    # Vertical bounds: from local tree to top of legend area
    y_min = local_tree_top + LEGEND_TREE_GAP
    y_max = legend_top_y

    # Ensure valid region
    if x_max <= x_min or y_max <= y_min:
        return BoundingBox(0, 0, 0, 0)  # Zero-area box

    return BoundingBox(x_min, y_min, x_max, y_max)
