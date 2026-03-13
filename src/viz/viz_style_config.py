"""Centralized visualization style configuration.

Edit this file to change colors and styling across ALL visualizations.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMICS PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# Dynamics metric colors
DYNAMICS_COLORS = {
    "pull": "#E67E22",      # Orange - normative strength
    "drift": "#8E44AD",     # Purple - deviation from initial
    "horizon": "#2980B9",   # Blue - distance to final
}

# Dynamics line/marker styling
DYNAMICS_LINE_WIDTH = 2.0
DYNAMICS_MARKER_SIZE = 6
DYNAMICS_MARKER_EDGE_WIDTH = 1.5

# ══════════════════════════════════════════════════════════════════════════════
# ARM COLORS
# ══════════════════════════════════════════════════════════════════════════════

# Colors for different arm types in estimation/tree plots
ARM_COLORS = {
    "root": "#95A5A6",      # Gray
    "trunk": "#E67E22",     # Orange
    "branch": "#3498DB",    # Blue
    "twig": "#27AE60",      # Green
}

# Extended arm colors for specific branches
ARM_COLORS_EXTENDED = {
    "root": "#95A5A6",
    "trunk": "#E67E22",
    "branch_1": "#3498DB",
    "branch_2": "#9B59B6",
    "twig_1_b1": "#27AE60",
    "twig_2_b1": "#2ECC71",
    "twig_1_b2": "#16A085",
    "twig_2_b2": "#1ABC9C",
}

# ══════════════════════════════════════════════════════════════════════════════
# BAR CHART COLORS
# ══════════════════════════════════════════════════════════════════════════════

# Core estimate bar colors
CORE_BAR_COLORS = {
    "positive": "#27AE60",  # Green - high compliance
    "negative": "#E74C3C",  # Red - low compliance
    "neutral": "#3498DB",   # Blue - moderate
}

# Deviance bar colors
DEVIANCE_BAR_COLORS = {
    "trunk": "#E67E22",
    "root": "#95A5A6",
}

# ══════════════════════════════════════════════════════════════════════════════
# GRID STYLING
# ══════════════════════════════════════════════════════════════════════════════

# Major grid
GRID_ALPHA_MAJOR = 0.5
GRID_LINE_WIDTH_MAJOR = 0.6

# Minor grid
GRID_ALPHA_MINOR = 0.25
GRID_LINE_WIDTH_MINOR = 0.4

# ══════════════════════════════════════════════════════════════════════════════
# FONT SIZES
# ══════════════════════════════════════════════════════════════════════════════

TITLE_FONTSIZE = 12
TITLE_FONTWEIGHT = "bold"
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 9
LEGEND_FONTSIZE = 9

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Default figure sizes (width, height)
FIGURE_SIZE_DYNAMICS = (12, 7)
FIGURE_SIZE_BARPLOT = (10, 6)
FIGURE_SIZE_TREE = (14, 10)

# Export settings
DPI = 150
FACECOLOR = "white"

# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON CHART COLORS
# ══════════════════════════════════════════════════════════════════════════════

# Method comparison colors (for comparing generation methods)
METHOD_COLORS = {
    "simple-sampling": "#3498DB",
    "forking-paths": "#E67E22",
    "seeking-entropy": "#9B59B6",
    "just-greedy": "#95A5A6",
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════


def get_arm_color(arm_name: str) -> str:
    """Get color for an arm by name."""
    # Check extended colors first
    if arm_name in ARM_COLORS_EXTENDED:
        return ARM_COLORS_EXTENDED[arm_name]

    # Fall back to arm type
    if arm_name == "root":
        return ARM_COLORS["root"]
    elif arm_name == "trunk":
        return ARM_COLORS["trunk"]
    elif arm_name.startswith("branch"):
        return ARM_COLORS["branch"]
    elif arm_name.startswith("twig"):
        return ARM_COLORS["twig"]

    return "#1E90FF"  # Default blue


def get_dynamics_color(metric: str) -> str:
    """Get color for a dynamics metric."""
    return DYNAMICS_COLORS.get(metric, "#1E90FF")


def get_method_color(method: str) -> str:
    """Get color for a generation method."""
    return METHOD_COLORS.get(method, "#1E90FF")
