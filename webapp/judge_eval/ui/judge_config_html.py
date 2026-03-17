"""Judge page HTML configuration panel."""

from __future__ import annotations

from webapp.app_settings import DEFAULT_PROMPT, DEFAULT_QUESTIONS


def get_judge_config_html() -> str:
    # Escape braces for f-string since judge prompt contains {text} and {question}
    prompt_escaped = DEFAULT_PROMPT.replace("{", "{{").replace("}", "}}")
    # Escape newlines for HTML attributes
    questions_escaped = DEFAULT_QUESTIONS.replace("\n", "&#10;")
    return f"""
<div class="config-panel" id="judgeConfig">
    <button class="config-back" onclick="showLanding()">&larr; Back</button>
    <div class="config-card" style="max-width:900px">
        <h2 class="config-title">Judge</h2>
        <div class="config-row" style="gap:15px;margin-bottom:15px">
            <div class="config-group" style="flex:1"><div class="config-label">Provider</div><select class="config-input" id="judgeProvider" onchange="updateJudgeConfigModels()"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="huggingface">HuggingFace (Local)</option></select></div>
            <div class="config-group" style="flex:2"><div class="config-label">Model</div><select class="config-input" id="judgeModel" onchange="onJudgeModelChange()"></select></div>
        </div>
        <div class="config-row" style="gap:20px;align-items:stretch">
            <div class="config-group" style="flex:3;display:flex;flex-direction:column">
                <div class="config-label">Prompt Format</div>
                <div class="syntax-editor" id="judgePromptFormat" data-init="{prompt_escaped}" style="flex:1;min-height:220px"></div>
                <div class="config-hint">Placeholders: <span style="color:var(--cyan)">{{text}}</span> and <span style="color:var(--pink)">{{question}}</span></div>
            </div>
            <div class="config-group" style="flex:2;display:flex;flex-direction:column">
                <div class="config-label">Questions <span style="font-weight:normal;opacity:0.6">(colors match legend)</span></div>
                <div class="colored-editor" id="judgeQuestions" data-init="{questions_escaped}" data-placeholder="Enter a question..." style="flex:1;min-height:220px"></div>
            </div>
        </div>
        <button class="start-btn secondary" onclick="startJudge()">Start</button>
        <div class="config-hint" style="text-align:center;margin-top:10px">You'll add texts to evaluate on the next screen.</div>
    </div>
</div>
"""
