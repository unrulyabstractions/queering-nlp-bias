"""Shared HTML components: landing, settings, controls."""

from __future__ import annotations

from webapp.app_settings import DEFAULT_JUDGE_PROMPT, DEFAULT_JUDGE_TEXT


def get_landing_html() -> str:
    return """
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
<style>.hero-title, .mode-card .mode-name, .config-title { font-family: 'Roboto Mono', monospace !important; }</style>
<div class="landing" id="landing">
    <div class="landing-header">
        <img class="landing-logo" src="https://images.squarespace-cdn.com/content/v1/628d3c30b420d16dfbab5863/dfab906c-7535-4b9b-a4bf-fd6869226dba/QiAI_lateral_RGB_1200x284.png" alt="Queer in AI">
        <button class="landing-settings" onclick="showSettings()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
        </button>
    </div>
    <div class="landing-content">
        <div class="mode-grid">
            <div class="mode-card mode-card-large mode-card-forking" onclick="selectMode('tree')">
                <div class="mode-card-banner" style="background-image: url('/static/forking_banner.png')"></div>
                <div class="mode-card-overlay">
                    <div class="mode-name">Forking and Localizing Normativity</div>
                </div>
            </div>
            <div class="mode-card mode-card-large mode-card-dynamics" onclick="selectMode('dynamics')">
                <div class="mode-card-banner" style="background-image: url('/static/dynamics_banner.png')"></div>
                <div class="mode-card-overlay">
                    <div class="mode-name">Dynamics of Meaning</div>
                </div>
            </div>
            <div class="mode-card mode-card-small" onclick="selectMode('judge')">
                <div class="mode-name">Judge</div>
                <div class="mode-desc">Evaluate Judge LLM</div>
            </div>
        </div>
        <div class="landing-hero">
            <a href="https://github.com/unrulyabstractions/queering-nlp-bias" target="_blank" class="hero-link">
                <h1 class="hero-title">Queering<br>NLP<br>Bias</h1>
            </a>
        </div>
    </div>
</div>
"""


def get_settings_html() -> str:
    # Escape braces for f-string since judge prompt contains {text} and {question}
    prompt_escaped = DEFAULT_JUDGE_PROMPT.replace("{", "{{").replace("}", "}}")
    return f"""
<div class="config-panel" id="settingsModal">
    <button class="config-back" onclick="hideSettings()">&larr; Back</button>
    <div class="config-card" style="max-width:1000px">
        <h2 class="config-title">Settings</h2>
        <div class="settings-columns">
            <div class="settings-left">
                <div class="config-row">
                    <div class="config-group"><div class="config-label">Generation Provider</div><select class="config-input" id="settingsGenProvider" onchange="updateGenModels()"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="huggingface">HuggingFace (Local)</option></select></div>
                    <div class="config-group"><div class="config-label">Generation Model</div><select class="config-input" id="settingsGenModel"></select></div>
                </div>
                <div class="config-row">
                    <div class="config-group"><div class="config-label">Judge Provider</div><select class="config-input" id="settingsJudgeProvider" onchange="updateJudgeModels()"><option value="openai">OpenAI</option><option value="anthropic">Anthropic</option><option value="huggingface">HuggingFace (Local)</option></select></div>
                    <div class="config-group"><div class="config-label">Judge Model</div><select class="config-input" id="settingsJudgeModel"></select></div>
                </div>
                <div class="config-row">
                    <div class="config-group"><div class="config-label">Temperature: <span id="tempVal">1.0</span></div><input type="range" class="config-input" id="settingsTemp" min="0" max="1" step="0.1" value="1.0" style="padding:0" oninput="document.getElementById('tempVal').textContent=this.value"></div>
                    <div class="config-group"><div class="config-label">Max Tokens: <span id="maxTokensVal">300</span></div><input type="range" class="config-input" id="settingsMaxTokens" min="50" max="1000" step="50" value="300" style="padding:0" oninput="document.getElementById('maxTokensVal').textContent=this.value"></div>
                </div>
                <div class="config-row">
                    <div class="config-group"><div class="config-label">Anthropic Key</div><input type="password" class="config-input" id="settingsAnthropicKey" placeholder="sk-ant-..."></div>
                    <div class="config-group"><div class="config-label">OpenAI Key</div><input type="password" class="config-input" id="settingsOpenaiKey" placeholder="sk-..."></div>
                </div>
            </div>
            <div class="settings-right" id="settingsJudgePromptSection">
                <div class="config-group" style="height:100%;display:flex;flex-direction:column">
                    <div class="config-label">Judge Prompt <span style="font-weight:normal;opacity:0.6">(for Forking &amp; Dynamics modes)</span></div>
                    <div class="syntax-editor" id="settingsJudgePrompt" data-init="{prompt_escaped}" style="flex:1;min-height:280px"></div>
                    <div class="config-hint">Placeholders: <span style="color:var(--cyan)">{{text}}</span> and <span style="color:var(--pink)">{{question}}</span>. Not used in Judge mode.</div>
                </div>
            </div>
        </div>
        <div class="config-row" style="gap:10px;margin-top:20px">
            <button class="start-btn" style="flex:1;margin-top:0" onclick="saveSettings()">Save</button>
            <button class="start-btn gold" style="flex:1;margin-top:0" onclick="resetSettings()">Reset</button>
        </div>
    </div>
</div>
"""


def get_controls_html() -> str:
    return f"""
<div class="progress-bar" id="progressBar" style="width: 0%;"></div>
<svg id="canvas"></svg>

<div class="floating-stats" id="floatingStats">
    <div class="stat-card"><div class="stat-value" id="statValue1">0</div><div class="stat-label" id="statLabel1">Samples</div></div>
    <div class="stat-card"><div class="stat-value" id="statValue2">0</div><div class="stat-label" id="statLabel2">Arms</div></div>
    <div class="live-badge" id="liveBadge"><span class="live-dot"></span><span class="live-text" id="liveText">Sampling</span></div>
</div>

<div class="mode-toggle" id="modeToggle"><button class="mode-btn active" id="btnCore">Core</button><button class="mode-btn" id="btnOrient">Orientation</button></div>
<div class="view-toggle" id="viewToggle"><button class="mode-btn active" id="btnEvolution">Evolution</button><button class="mode-btn" id="btnMagnitudes">Magnitudes</button></div>
<div class="evolution-mode-toggle" id="evolutionModeToggle"><button class="mode-btn small active" id="btnEvolutionCore">Core</button><button class="mode-btn small" id="btnEvolutionDrift">Drift</button><button class="mode-btn small" id="btnEvolutionPotential">Potential</button></div>
<div class="diversity-toggle" id="diversityToggle"><label class="toggle-label"><input type="checkbox" id="diversityCheckbox" onchange="toggleDiversity()"><span class="toggle-slider"></span><span class="toggle-text">Diversity</span></label></div>
<div class="legend" id="legend"></div>
<div class="control-buttons" id="controlButtons">
    <button class="control-btn back" onclick="goBack()">&larr; Back</button>
    <button class="control-btn stop" onclick="stopSampling()">Stop</button>
    <button class="control-btn reset" onclick="resetExperiment()">Reset</button>
    <button class="control-btn flip" id="flipAxesBtn" onclick="flipAxes()" style="display:none">Flip Axes</button>
</div>
<div class="zoom-controls" id="zoomControls">
    <button class="zoom-btn" onclick="zoomIn()" title="Zoom In">+</button>
    <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out">−</button>
    <button class="zoom-btn" onclick="zoomFit()" title="Fit to View">⊡</button>
    <button class="zoom-btn" onclick="zoomReset()" title="Reset Zoom">↺</button>
</div>
<div class="text-display" id="textDisplay"><div class="text-display-label">Text</div><div class="text-display-content" id="textContent"></div></div>
<div class="pie-chart-panel" id="pieChartPanel">
    <svg id="pieChart" width="140" height="140"></svg>
</div>
<div class="add-texts-panel" id="addTextsPanel">
    <div class="config-label">Texts to Judge <span style="font-weight:normal;opacity:0.6">(one per line)</span></div>
    <div class="colored-editor" id="addTextsInput" data-init="{DEFAULT_JUDGE_TEXT}" data-placeholder="Enter text to judge..." data-color-offset="4" style="min-height:150px"></div>
    <button class="start-btn add-texts-btn secondary" onclick="addMoreTexts()">&#10024; Judge Texts</button>
</div>
<div class="tooltip" id="tooltip"></div>
<div class="toast-container" id="toastContainer"></div>
<div class="trajectories-panel" id="trajectoriesPanel">
    <div class="trajectories-header">
        <span class="trajectories-title" id="trajectoriesTitle">Trajectories</span>
        <button class="trajectories-close" onclick="hideTrajectories()">&times;</button>
    </div>
    <div class="trajectories-prefix" id="trajectoriesPrefix"></div>
    <div class="trajectories-list" id="trajectoriesList"></div>
</div>

<div class="traj-explorer" id="trajExplorer">
    <button class="traj-explorer-close" onclick="hideTrajExplorer()">&times;</button>
    <div class="traj-explorer-hint">&#9650;&#9660; Navigate &bull; 1/2/3 Switch Mode &bull; Esc Close</div>
    <div class="traj-explorer-content">
        <div class="traj-explorer-text-section">
            <div class="traj-explorer-prompt" id="trajExplorerPrompt"></div>
            <div class="traj-explorer-text" id="trajExplorerText"></div>
            <div class="traj-explorer-slider-track" id="trajSliderTrack">
                <div class="traj-explorer-slider-ball" id="trajSliderBall"></div>
            </div>
            <div class="traj-explorer-diversity" id="trajExplorerDiversity">🌈 Diversity: <span id="trajDiversityValue">1.00</span></div>
        </div>
        <div class="traj-explorer-chart-section">
            <div class="traj-explorer-mode-toggle">
                <button class="traj-mode-btn active" id="trajModeCore" onclick="setTrajMode('core')">Core</button>
                <button class="traj-mode-btn" id="trajModeDrift" onclick="setTrajMode('drift')">Drift</button>
                <button class="traj-mode-btn" id="trajModePotential" onclick="setTrajMode('potential')">Potential</button>
            </div>
            <svg class="traj-explorer-chart" id="trajExplorerChart"></svg>
            <div class="traj-explorer-legend" id="trajExplorerLegend"></div>
        </div>
    </div>
</div>
"""
