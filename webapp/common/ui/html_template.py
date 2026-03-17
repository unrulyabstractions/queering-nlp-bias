"""HTML/CSS/JS template assembler - combines all page components."""

from __future__ import annotations

from webapp.common.ui.core_math_js import get_core_math_js
from webapp.common.ui.page_styles import get_css
from webapp.common.ui.shared_app_js import get_shared_app_js
from webapp.common.ui.shared_components import (
    get_controls_html,
    get_landing_html,
    get_settings_html,
)
from webapp.common.ui.ui_text_config import APP_TITLE
from webapp.dynamics_analysis.ui import get_dynamics_config_html, get_dynamics_page_js
from webapp.judge_eval.ui import get_judge_config_html, get_judge_page_js
from webapp.tree_exploration.ui import get_tree_config_html, get_tree_page_js


def get_html_template() -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{APP_TITLE}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>{get_css()}</style>
</head>
<body>
    {get_controls_html()}
    {get_landing_html()}
    {get_settings_html()}
    {get_tree_config_html()}
    {get_dynamics_config_html()}
    {get_judge_config_html()}
    <script>{get_javascript()}</script>
</body>
</html>"""


def get_javascript() -> str:
    return f"""
{get_core_math_js()}
{get_shared_app_js()}
{get_tree_page_js()}
{get_dynamics_page_js()}
{get_judge_page_js()}

// ════════════════════════════════════════════════════════════════════════════════
// WINDOW RESIZE HANDLER
// ════════════════════════════════════════════════════════════════════════════════
window.addEventListener('resize',()=>{{
    if(currentMode==='tree'&&treeState.nodes.length)initTreeViz();
    if(currentMode==='dynamics'&&dynamicsState.positions.length)drawDynamicsChart();
    if(currentMode==='judge'&&judgeState.results.length)drawJudgeHeatmap();
}});
"""
