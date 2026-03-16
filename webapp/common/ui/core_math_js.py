"""CoreMath JavaScript module - display helpers only, NO calculations."""

from __future__ import annotations


def get_core_math_js() -> str:
    return """
// ════════════════════════════════════════════════════════════════════════════════
// CORE MATH MODULE - Display helpers only. All calculations done server-side.
// ════════════════════════════════════════════════════════════════════════════════
const CoreMath = {
    // Bar geometry for visualization - uses pre-computed orientation from server
    computeBarGeometry(score, orientation, barWidth, showOrientation) {
        if (showOrientation && orientation !== null && orientation !== undefined) {
            return {
                width: Math.abs(orientation) * barWidth,
                x: orientation >= 0 ? 0 : -Math.abs(orientation) * barWidth,
                value: orientation
            };
        }
        return { width: score * barWidth, x: 0, value: score };
    }
};
"""
