"""Dynamics page HTML configuration panel."""

from __future__ import annotations

from webapp.app_settings import DEFAULT_DYN_TRAJ, DEFAULT_PROMPT, DEFAULT_QUESTIONS


def get_dynamics_config_html() -> str:
    # Escape newlines for HTML attributes
    questions_escaped = DEFAULT_QUESTIONS.replace("\n", "&#10;")
    return f"""
<div class="config-panel config-panel-dynamics" id="dynamicsConfig">
    <div class="config-panel-bg" style="background-image: url('/static/dynamics_banner.png')"></div>
    <button class="config-back" onclick="showLanding()">&larr; Back</button>
    <div class="config-card" style="max-width:900px">
        <h2 class="config-title">Dynamics of Meaning</h2>
        <div class="config-row" style="gap:20px">
            <div class="config-group" style="flex:1">
                <div class="config-label">Prompt</div>
                <textarea class="config-textarea" id="dynPrompt" style="min-height:120px">{DEFAULT_PROMPT}</textarea>
                <div class="config-label" style="margin-top:15px">Continuation <span style="font-weight:normal;opacity:0.6">(leave empty to generate)</span></div>
                <textarea class="config-textarea" id="dynContinuation" style="min-height:80px">{DEFAULT_DYN_TRAJ}</textarea>
            </div>
            <div class="config-group" style="flex:1">
                <div class="config-label">Questions <span style="font-weight:normal;opacity:0.6">(colors match legend)</span></div>
                <div class="colored-editor" id="dynQuestions" data-init="{questions_escaped}" data-placeholder="Enter a question..." style="min-height:220px"></div>
            </div>
        </div>
        <div class="config-group" style="margin-top:20px"><div class="config-label">Samples per position: <span id="dynStepVal">1</span></div><input type="range" class="config-input" id="dynStep" min="1" max="100" value="1" style="padding:0" oninput="document.getElementById('dynStepVal').textContent=this.value"></div>
        <button class="start-btn" onclick="startDynamics()">Start</button>
    </div>
</div>
"""
