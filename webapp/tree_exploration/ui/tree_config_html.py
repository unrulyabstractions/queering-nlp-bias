"""Tree page HTML configuration panel."""

from __future__ import annotations

from webapp.app_settings import (
    DEFAULT_PREFIXES,
    DEFAULT_PROMPT,
    DEFAULT_QUESTIONS,
    DEFAULT_SAMPLES_PER_NODE,
)


def get_tree_config_html() -> str:
    # Escape newlines for HTML attributes (they get collapsed to spaces otherwise)
    prefixes_escaped = DEFAULT_PREFIXES.replace("\n", "&#10;")
    questions_escaped = DEFAULT_QUESTIONS.replace("\n", "&#10;")
    return f"""
<div class="config-panel config-panel-forking" id="treeConfig">
    <div class="config-panel-bg" style="background-image: url('/static/forking_banner.png')"></div>
    <button class="config-back" onclick="showLanding()">&larr; Back</button>
    <div class="config-card" style="max-width:1100px">
        <h2 class="config-title">Forking and Localizing Normativity</h2>
        <div class="config-row" style="gap:20px;align-items:stretch">
            <div class="config-group" style="flex:1;display:flex;flex-direction:column">
                <div class="config-label">Prompt</div>
                <textarea class="config-textarea" id="treePrompt" style="flex:1;min-height:200px">{DEFAULT_PROMPT}</textarea>
            </div>
            <div class="config-group" style="flex:1;display:flex;flex-direction:column">
                <div class="config-label">Prefixes <span style="font-weight:normal;opacity:0.6">(one per line)</span></div>
                <div class="colored-editor" id="treePrefixes" data-init="{prefixes_escaped}" data-placeholder="Enter a prefix..." data-color-offset="3" style="flex:1;min-height:200px"></div>
            </div>
            <div class="config-group" style="flex:1;display:flex;flex-direction:column">
                <div class="config-label">Questions <span style="font-weight:normal;opacity:0.6">(colors match legend)</span></div>
                <div class="colored-editor" id="treeQuestions" data-init="{questions_escaped}" data-placeholder="Enter a question..." style="flex:1;min-height:200px"></div>
            </div>
        </div>
        <div class="config-group" style="margin-top:20px"><div class="config-label">Samples per node: <span id="treeRoundsVal">{DEFAULT_SAMPLES_PER_NODE}</span></div><input type="range" class="config-input" id="treeRounds" min="1" max="500" value="{DEFAULT_SAMPLES_PER_NODE}" style="padding:0" oninput="document.getElementById('treeRoundsVal').textContent=this.value"></div>
        <button class="start-btn" onclick="startTree()">Start</button>
    </div>
</div>
"""
