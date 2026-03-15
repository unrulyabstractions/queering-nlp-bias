"""Bounding box and collision detection utilities for visualization.

Provides axis-aligned bounding boxes and content tracking for legend placement
optimization. All coordinates are in matplotlib data units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box for collision detection.

    All coordinates in data units. Origin at bottom-left.
    Immutable (frozen) for safe use in sets and as dict keys.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        """Box width in data units."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Box height in data units."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """Box area in squared data units."""
        return self.width * self.height

    @property
    def center_x(self) -> float:
        """X coordinate of box center."""
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        """Y coordinate of box center."""
        return (self.y_min + self.y_max) / 2

    def intersects(self, other: BoundingBox) -> bool:
        """Check if this box overlaps with another."""
        return not (
            self.x_max < other.x_min
            or self.x_min > other.x_max
            or self.y_max < other.y_min
            or self.y_min > other.y_max
        )

    def intersection_area(self, other: BoundingBox) -> float:
        """Calculate area of intersection with another box."""
        if not self.intersects(other):
            return 0.0

        x_overlap = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        y_overlap = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        return max(0, x_overlap) * max(0, y_overlap)

    def distance_to(self, other: BoundingBox) -> float:
        """Compute minimum distance between two boxes.

        Returns 0 if boxes overlap or touch.
        """
        dx = max(0, max(self.x_min - other.x_max, other.x_min - self.x_max))
        dy = max(0, max(self.y_min - other.y_max, other.y_min - self.y_max))
        return (dx**2 + dy**2) ** 0.5

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside this box."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def expand(self, margin: float) -> BoundingBox:
        """Return a new box expanded by margin on all sides."""
        return BoundingBox(
            self.x_min - margin,
            self.y_min - margin,
            self.x_max + margin,
            self.y_max + margin,
        )

    @classmethod
    def from_center(
        cls, cx: float, cy: float, width: float, height: float
    ) -> BoundingBox:
        """Create box from center point and dimensions."""
        return cls(
            cx - width / 2,
            cy - height / 2,
            cx + width / 2,
            cy + height / 2,
        )

    @classmethod
    def union(cls, boxes: list[BoundingBox]) -> BoundingBox | None:
        """Create bounding box that contains all given boxes."""
        if not boxes:
            return None
        return cls(
            min(b.x_min for b in boxes),
            min(b.y_min for b in boxes),
            max(b.x_max for b in boxes),
            max(b.y_max for b in boxes),
        )


class TreeContentTracker:
    """Track bounding boxes of all tree content for collision detection.

    Collects bounds during tree rendering, then provides collision queries.
    Used by legend placement optimization to avoid overlapping tree content.
    """

    def __init__(self) -> None:
        self.node_boxes: list[BoundingBox] = []
        self.node_labels: list[BoundingBox] = []
        self.edges: list[BoundingBox] = []
        self.title_box: BoundingBox | None = None

    def add_node_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        """Add a node's background box (bar chart container)."""
        self.node_boxes.append(
            BoundingBox(x, y - height / 2, x + width, y + height / 2)
        )

    def add_node_label(
        self,
        x: float,
        y: float,
        text: str,
        fontsize: float,
        char_width: float = 0.05,
    ) -> None:
        """Add a node's label text bounding box."""
        text_width = len(text) * char_width * (fontsize / 10)
        text_height = char_width * 1.5 * (fontsize / 10)
        self.node_labels.append(BoundingBox(x, y, x + text_width, y + text_height))

    def add_edge(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        line_width: float = 0.1,
    ) -> None:
        """Add an edge's bounding box."""
        margin = line_width / 2
        self.edges.append(
            BoundingBox(
                min(x1, x2) - margin,
                min(y1, y2) - margin,
                max(x1, x2) + margin,
                max(y1, y2) + margin,
            )
        )

    def set_title(self, x: float, y: float, width: float, height: float) -> None:
        """Set the title bounding box."""
        self.title_box = BoundingBox(x, y, x + width, y + height)

    def get_all_content_boxes(self) -> list[BoundingBox]:
        """Get all content bounding boxes including title."""
        boxes = self.node_boxes + self.node_labels + self.edges
        if self.title_box:
            boxes.append(self.title_box)
        return boxes

    def get_content_bounds(self) -> BoundingBox | None:
        """Get overall bounding box of all content."""
        return BoundingBox.union(self.get_all_content_boxes())

    def get_tree_bounds(self) -> BoundingBox | None:
        """Get bounding box of tree nodes only (no labels/title)."""
        return BoundingBox.union(self.node_boxes)

    def get_content_in_x_range(
        self, x_min: float, x_max: float, *, mostly_contained: bool = False
    ) -> list[BoundingBox]:
        """Get all content boxes that overlap with the given x range.

        Args:
            x_min, x_max: X range to check
            mostly_contained: If True, only include boxes whose CENTER is in range
        """
        result = []
        for box in self.node_boxes + self.node_labels:
            if mostly_contained:
                box_center_x = (box.x_min + box.x_max) / 2
                if x_min <= box_center_x <= x_max:
                    result.append(box)
            else:
                if box.x_max > x_min and box.x_min < x_max:
                    result.append(box)
        return result

    def get_max_y_in_x_range(
        self, x_min: float, x_max: float, *, mostly_contained: bool = True
    ) -> float | None:
        """Get the maximum y coordinate of content centered in the given x range.

        Returns None if no content is centered in that range.
        """
        boxes = self.get_content_in_x_range(
            x_min, x_max, mostly_contained=mostly_contained
        )
        if not boxes:
            return None
        return max(box.y_max for box in boxes)

    def get_min_x_in_y_range(self, y_min: float, y_max: float) -> float | None:
        """Get the minimum x coordinate of any content in the given y range.

        Used to constrain legend width - legend shouldn't extend past
        the leftmost tree content at the same vertical level.
        """
        result = []
        for box in self.node_boxes + self.node_labels:
            if box.y_max > y_min and box.y_min < y_max:
                result.append(box.x_min)
        if not result:
            return None
        return min(result)


def compute_collision_score(
    legend_bounds: list[BoundingBox],
    tree_content: TreeContentTracker,
    collision_penalty: float = 10.0,
    *,
    include_edges: bool = False,
) -> float:
    """Compute collision penalty score.

    Returns 0 if no collisions, positive value proportional to overlap area.

    Args:
        legend_bounds: Bounding boxes of legend items
        tree_content: Tracker containing all tree element bounds
        collision_penalty: Multiplier for overlap area
        include_edges: If False (default), only check against node boxes and
            labels, not edges. Edges are thin lines that render behind legend.
    """
    if include_edges:
        content_boxes = tree_content.get_all_content_boxes()
    else:
        content_boxes = list(tree_content.node_boxes) + list(tree_content.node_labels)
        if tree_content.title_box:
            content_boxes.append(tree_content.title_box)

    total_overlap = 0.0
    for legend_box in legend_bounds:
        for content_box in content_boxes:
            overlap = legend_box.intersection_area(content_box)
            total_overlap += overlap

    return total_overlap * collision_penalty


def compute_coverage_score(
    legend_bounds: list[BoundingBox],
    target_region: BoundingBox,
) -> float:
    """Compute coverage score - how much of target region is filled.

    Returns ratio of legend area to target region area (0.0 to 1.0+).
    """
    if target_region.area <= 0:
        return 0.0

    total_coverage = 0.0
    for legend_box in legend_bounds:
        coverage = legend_box.intersection_area(target_region)
        total_coverage += coverage

    return total_coverage / target_region.area


def compute_legend_bounds(
    layout: dict[str, Any],
    legend_top_y: float,
) -> list[BoundingBox]:
    """Convert legend layout to list of bounding boxes.

    Args:
        layout: Layout dict from compute_legend_layout()
        legend_top_y: Y coordinate for top of legend in data units

    Returns:
        List of BoundingBox for each legend item
    """
    bounds = []
    swatch_size = layout["swatch_size"]
    char_width = layout.get("char_width", 0.055)
    row_height = layout.get("row_height", 0.32)

    for item in layout["items"]:
        # Use wrapped line width, not full description width
        if "lines" in item and item["lines"]:
            # Max width of wrapped lines
            max_line_width = max(
                _estimate_text_width(line, char_width)
                for line in item["lines"]
            )
        else:
            # Fallback to full description
            max_line_width = _estimate_text_width(item["description"], char_width)

        item_x_min = item["swatch_x"]
        item_x_max = item["text_x"] + max_line_width

        item_y = legend_top_y + item["swatch_y"]
        half_height = max(swatch_size, row_height) / 2
        item_y_min = item_y - half_height
        item_y_max = item_y + half_height

        bounds.append(BoundingBox(item_x_min, item_y_min, item_x_max, item_y_max))

    return bounds


# Text width estimation constants
_WIDE_CHARS = frozenset("MWmw@%#&QDOGHUB")
_NARROW_CHARS = frozenset("il1!|,.:;'`()[]{}fjrt ")


def _estimate_text_width(text: str, char_width: float) -> float:
    """Estimate text width with character-aware calculation.

    Wide chars (M, W, etc): 1.4x base width
    Narrow chars (i, l, etc): 0.5x base width
    """
    total = 0.0
    for c in text:
        if c in _WIDE_CHARS:
            total += char_width * 1.4
        elif c in _NARROW_CHARS:
            total += char_width * 0.5
        else:
            total += char_width
    return total
