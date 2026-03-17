"""Judge page HTML configuration panel."""

from __future__ import annotations

from webapp.app_settings import DEFAULT_QUESTIONS
from webapp.common.ui.ui_text_config import MODE_TITLE_JUDGE_CONFIG


def get_judge_config_html() -> str:
    # Escape newlines for HTML attributes
    questions_escaped = DEFAULT_QUESTIONS.replace("\n", "&#10;")
    return f"""
<div class="config-panel" id="judgeConfig">
    <button class="config-back" onclick="showLanding()"><span class="back-arrow">←</span><span class="back-text">BACK</span></button>
    <div class="config-card" style="max-width:900px">
        <h2 class="config-title">{MODE_TITLE_JUDGE_CONFIG}</h2>
        <div class="config-row" style="gap:15px;margin-bottom:15px">
            <div class="config-group" style="flex:3"><div class="config-label">Judge Models <span style="font-weight:normal;opacity:0.6">(select from any provider)</span></div><div class="model-checkbox-list" id="judgeModelList" style="max-height:120px"></div></div>
            <div class="config-group" style="flex:1"><div class="config-label">Temperature: <span id="judgeTempValConfig">0.0</span></div><input type="range" class="config-input" id="judgeTemperatureConfig" min="0" max="1" step="0.1" value="0.0" style="padding:0" oninput="document.getElementById('judgeTempValConfig').textContent=this.value"></div>
            <div class="config-group" style="flex:1"><div class="config-label">Max Tokens: <span id="judgeMaxTokensVal">32</span></div><input type="range" class="config-input" id="judgeMaxTokensConfig" min="2" max="2000" step="2" value="32" style="padding:0" oninput="document.getElementById('judgeMaxTokensVal').textContent=this.value"></div>
        </div>
        <div class="config-row" style="gap:20px;align-items:stretch">
            <div class="config-group" style="flex:3;display:flex;flex-direction:column">
                <div class="config-label">Prompt Format</div>
                <div class="syntax-editor" id="judgePromptFormat" style="flex:1;min-height:220px"></div>
                <div class="config-hint">Placeholders: <span style="color:var(--cyan)">{{text}}</span> and <span style="color:var(--pink)">{{question}}</span></div>
            </div>
            <div class="config-group" style="flex:2;display:flex;flex-direction:column">
                <div class="config-label">Questions <span style="font-weight:normal;opacity:0.6">(colors match legend)</span></div>
                <div class="colored-editor" id="judgeQuestions" data-init="{questions_escaped}" data-placeholder="Enter a question..." style="flex:1;min-height:220px"></div>
            </div>
        </div>
        <button class="start-btn secondary" onclick="startJudge()">Start</button>
        <div class="config-hint" style="text-align:center;margin-top:10px">You'll add texts to evaluate on the next screen.</div>
        <button style="display:block;margin:15px auto 0;padding:6px 16px;background:transparent;border:1px solid var(--border);color:var(--text-dim);border-radius:6px;font-size:12px;cursor:pointer" onclick="resetJudgeSettings()">Reset to Defaults</button>
    </div>
</div>
"""
