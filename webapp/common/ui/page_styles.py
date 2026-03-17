"""Shared CSS styles for all pages."""

from __future__ import annotations


def get_css() -> str:
    return """
* { margin: 0; padding: 0; box-sizing: border-box; }
:root {
    --pink: #FF6B9D; --purple: #C678DD; --cyan: #56B6C2; --gold: #E5C07B;
    --mint: #98C379; --blue: #61AFEF; --coral: #E06C75; --orange: #D19A66;
    --bg-dark: #0a0a12; --bg-card: rgba(25,30,45,0.95);
    --text: #e0e0e0; --text-dim: #6b7280;
}
html, body { width: 100%; height: 100%; overflow: hidden; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
body { background: linear-gradient(135deg, #fef6f9 0%, #f8f4ff 30%, #f0f7ff 60%, #fff5f0 100%); color: #4a3f5c; }
@keyframes grad { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }

.landing { position: fixed; inset: 0; display: flex; flex-direction: column; z-index: 2000; transition: opacity .5s; padding: 40px 50px; background: linear-gradient(135deg, #fef6f9 0%, #f8f4ff 30%, #f0f7ff 60%, #fff5f0 100%); }
.landing.hidden { opacity: 0; pointer-events: none; }
.landing-header { display: flex; align-items: center; gap: 20px; margin-bottom: 30px; }
.landing-logo { height: 28px; object-fit: contain; }
.landing-settings { margin-left: auto; background: rgba(255,255,255,.7); border: 1px solid rgba(120,100,140,.15); border-radius: 20px; padding: 24px 30px; color: #7a6b8a; cursor: pointer; transition: all .2s; display: flex; align-items: center; justify-content: center; }
.landing-settings:hover { border-color: #9b7cb8; color: #9b7cb8; background: rgba(255,255,255,.9); transform: scale(1.05); }
.landing-settings svg { width: 56px; height: 56px; }
.landing-content { display: flex; flex: 1; gap: 40px; }
.mode-grid { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr auto; gap: 20px; flex: 0 0 auto; width: 700px; }
.mode-card { background: rgba(255,255,255,.85); border: 1px solid rgba(180,160,200,.2); border-radius: 20px; padding: 30px; cursor: pointer; transition: all .25s; display: flex; flex-direction: column; justify-content: flex-end; box-shadow: 0 4px 20px rgba(100,80,120,.06); }
.mode-card:hover { border-color: rgba(155,124,184,.4); background: rgba(255,255,255,.95); transform: translateY(-2px); box-shadow: 0 8px 30px rgba(100,80,120,.12); }
.mode-card-large { min-height: 220px; }
.mode-card-small { grid-column: 2; justify-self: end; width: 180px; min-height: 120px; padding: 20px; text-align: center; justify-content: center; background: linear-gradient(135deg, rgba(255,255,255,.9) 0%, rgba(248,240,255,.95) 50%, rgba(240,248,255,.9) 100%); border: 2px solid transparent; background-clip: padding-box; position: relative; }
.mode-card-small::before { content: ''; position: absolute; inset: -2px; border-radius: 22px; background: linear-gradient(135deg, #c490d1, #7ec8c8, #e5c07b); z-index: -1; opacity: 0.6; transition: opacity .3s; }
.mode-card-small:hover::before { opacity: 1; }
.mode-card-small:hover .mode-name { -webkit-text-fill-color: white; text-shadow: 0 0 20px rgba(255,255,255,.5); }
.mode-card .mode-name { font-family: 'Roboto Mono', 'SF Mono', 'Monaco', 'Consolas', monospace; font-size: 15px; font-weight: 700; color: #4a3f5c; margin-bottom: 10px; line-height: 1.3; }
.mode-card .mode-desc { font-size: 12px; color: #7a6b8a; line-height: 1.5; }
.mode-card-small .mode-name { font-family: 'Roboto Mono', 'SF Mono', 'Monaco', 'Consolas', monospace; font-size: 32px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; line-height: 1.15; background: linear-gradient(135deg, #9b7cb8, #5a9898); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0; }
.mode-card-small .mode-desc { font-size: 10px; color: #9a8baa; }

/* Cards with banners - queer aesthetic */
.mode-card-forking, .mode-card-dynamics { padding: 0; overflow: hidden; position: relative; }
.mode-card-forking .mode-card-banner, .mode-card-dynamics .mode-card-banner { position: absolute; inset: 0; background-size: cover; background-position: center; filter: saturate(1.3) contrast(1.05); transition: all .4s ease; }
.mode-card-forking:hover .mode-card-banner, .mode-card-dynamics:hover .mode-card-banner { filter: saturate(1.5) contrast(1.1) brightness(1.05); transform: scale(1.02); }
.mode-card-forking .mode-card-overlay, .mode-card-dynamics .mode-card-overlay { position: absolute; inset: 0; display: flex; justify-content: flex-start; padding: 20px; }
.mode-card-forking .mode-card-overlay { align-items: flex-start; }
.mode-card-dynamics .mode-card-overlay { align-items: flex-end; }
.mode-card-forking .mode-name, .mode-card-dynamics .mode-name { color: white; font-size: 32px; text-transform: uppercase; letter-spacing: 0.08em; line-height: 1.15; font-weight: 700; padding: 0; position: relative; z-index: 2; background: transparent; }
.mode-card-forking .mode-name { text-shadow: 0 0 10px rgba(255,255,255,1), 0 0 30px rgba(255,150,200,1), 0 0 60px rgba(200,100,220,1), 0 0 100px rgba(180,80,200,.8), 0 0 150px rgba(200,100,220,.6); }
.mode-card-dynamics .mode-name { text-shadow: 0 0 10px rgba(255,255,255,1), 0 0 30px rgba(150,220,255,1), 0 0 60px rgba(100,200,255,1), 0 0 100px rgba(80,180,220,.8), 0 0 150px rgba(100,200,255,.6); }
/* Amorphous blur blobs behind text - organic, queer aesthetic */
.mode-card-forking .mode-card-overlay::before, .mode-card-forking .mode-card-overlay::after,
.mode-card-dynamics .mode-card-overlay::before, .mode-card-dynamics .mode-card-overlay::after { content: ''; position: absolute; z-index: 1; filter: blur(30px); pointer-events: none; }
.mode-card-forking .mode-card-overlay::before { width: 280px; height: 160px; top: 5px; left: -20px; background: rgba(180,100,200,.85); border-radius: 65% 35% 50% 50% / 50% 55% 45% 50%; animation: blobPulse1 4s ease-in-out infinite; }
.mode-card-forking .mode-card-overlay::after { width: 240px; height: 140px; top: 40px; left: 80px; background: rgba(255,120,200,.75); border-radius: 40% 60% 35% 65% / 55% 40% 60% 45%; animation: blobPulse2 5s ease-in-out infinite; }
.mode-card-dynamics .mode-card-overlay::before { width: 260px; height: 150px; bottom: 5px; left: -15px; background: rgba(100,180,220,.85); border-radius: 50% 50% 55% 45% / 40% 60% 40% 60%; animation: blobPulse1 4.5s ease-in-out infinite; }
.mode-card-dynamics .mode-card-overlay::after { width: 220px; height: 130px; bottom: 35px; left: 100px; background: rgba(80,220,200,.75); border-radius: 55% 45% 40% 60% / 60% 40% 55% 45%; animation: blobPulse2 5.5s ease-in-out infinite; }
/* Third blob via mode-name pseudo-element */
.mode-card-forking .mode-name::before, .mode-card-dynamics .mode-name::before { content: ''; position: absolute; z-index: -1; filter: blur(35px); pointer-events: none; }
.mode-card-forking .mode-name::before { width: 200px; height: 120px; top: -30px; right: -40px; background: rgba(200,80,180,.7); border-radius: 35% 65% 55% 45% / 45% 50% 50% 55%; animation: blobPulse3 6s ease-in-out infinite; }
.mode-card-dynamics .mode-name::before { width: 140px; height: 75px; bottom: -25px; right: -30px; background: rgba(60,200,220,.4); border-radius: 45% 55% 50% 50% / 55% 45% 55% 45%; animation: blobPulse3 5s ease-in-out infinite; }
@keyframes blobPulse1 { 0%, 100% { transform: scale(1) translate(0, 0); } 50% { transform: scale(1.1) translate(5px, -3px); } }
@keyframes blobPulse2 { 0%, 100% { transform: scale(1) translate(0, 0); } 50% { transform: scale(1.05) translate(-8px, 5px); } }
@keyframes blobPulse3 { 0%, 100% { transform: scale(1) translate(0, 0); } 50% { transform: scale(1.15) translate(10px, -5px); } }

.landing-hero { flex: 1; display: flex; align-items: center; justify-content: center; }
.hero-link { text-decoration: none; display: inline-block; transition: transform 0.3s, filter 0.3s; }
.hero-link:hover { transform: scale(1.03); filter: drop-shadow(0 0 25px rgba(180,100,200,.4)); }
.hero-title { font-family: 'Roboto Mono', 'SF Mono', 'Monaco', 'Consolas', 'Liberation Mono', monospace; font-size: 88px; font-weight: 700; color: #4a3f5c; line-height: 1.05; text-align: right; opacity: 0.9; letter-spacing: -0.02em; }

.config-panel { position: fixed; inset: 0; background: linear-gradient(135deg, #fef6f9 0%, #f8f4ff 30%, #f0f7ff 60%, #fff5f0 100%); z-index: 2100; display: flex; align-items: center; justify-content: center; opacity: 0; visibility: hidden; transition: all .3s; }
.config-panel.visible { opacity: 1; visibility: visible; }

/* Config panels with crisp background */
.config-panel-forking, .config-panel-dynamics { background: transparent; }
.config-panel-forking .config-panel-bg, .config-panel-dynamics .config-panel-bg { position: absolute; inset: 0; background-size: cover; background-position: center; filter: saturate(1.2) brightness(1.05) contrast(1.02); z-index: -1; }
.config-panel-forking .config-card, .config-panel-dynamics .config-card { background: rgba(255,255,255,.65); backdrop-filter: blur(100px) saturate(2); border: none; border-radius: 50px; }
.config-card { background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.25); border-radius: 24px; padding: 40px; width: 90%; max-width: 700px; max-height: 90vh; overflow-y: auto; box-shadow: 0 8px 40px rgba(100,80,120,.1); }
.config-card.large { max-width: 850px; }
.config-title { font-family: 'Roboto Mono', monospace; font-size: 20px; font-weight: 700; text-align: center; margin-bottom: 30px; color: #4a3f5c; }
.config-back { position: absolute; top: 30px; left: 30px; background: rgba(255,255,255,.15); backdrop-filter: blur(10px); border: 1px solid rgba(255,107,157,.15); padding: 16px 28px; cursor: pointer; display: flex; align-items: center; gap: 12px; border-radius: 12px; }
.config-back .back-arrow { font-size: 18px; line-height: 1; color: rgba(255,107,157,.5); }
.config-back .back-text { font-family: 'Roboto Mono', monospace; font-size: 15px; font-weight: 600; letter-spacing: 0.06em; color: rgba(255,107,157,.5); }
.config-group { margin-bottom: 20px; }
.config-label { font-size: 11px; font-weight: 600; color: #7a6b8a; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 8px; }
.config-hint { font-size: 10px; color: #9a8baa; margin-top: 6px; font-style: italic; }
.config-input, .config-textarea { width: 100%; background: rgba(255,255,255,.8); border: 1px solid rgba(180,160,200,.3); border-radius: 10px; padding: 12px 16px; color: #4a3f5c; font-size: 14px; font-family: inherit; transition: all .2s; }
.config-input:focus, .config-textarea:focus { outline: none; border-color: #9b7cb8; background: rgba(255,255,255,.95); }
select.config-input { appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12'%3E%3Cpath fill='%237a6b8a' d='M6 8L2 4h8z'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 12px center; padding-right: 36px; cursor: pointer; }
select.config-input option { background: #fff; color: #4a3f5c; }
.config-textarea { min-height: 80px; resize: vertical; }
.config-textarea.tall { min-height: 150px; }
.config-row { display: flex; gap: 15px; }
.config-row .config-group { flex: 1; }
.start-btn { width: 100%; padding: 16px; margin-top: 20px; background: linear-gradient(135deg, #c490d1, #9b7cb8); border: none; border-radius: 12px; color: white; font-size: 14px; font-weight: 700; cursor: pointer; transition: all .3s; text-transform: uppercase; letter-spacing: .1em; font-family: 'Roboto Mono', monospace; }
.start-btn:hover { transform: translateY(-2px); box-shadow: 0 10px 40px rgba(155,124,184,.35); }
.start-btn.secondary { background: linear-gradient(135deg, #7ec8c8, #5ba8b8); }
.start-btn.gold { background: linear-gradient(135deg, #e0c090, #c8a060); }

.settings-columns { display: flex; gap: 30px; }
.settings-left { flex: 1; min-width: 320px; }
.settings-right { flex: 1; min-width: 320px; }
@media (max-width: 800px) { .settings-columns { flex-direction: column; } }

.colored-editor { width: 100%; background: rgba(255,255,255,.8); border: 1px solid rgba(180,160,200,.3); border-radius: 10px; padding: 8px 4px; color: #4a3f5c; font-size: 13px; font-family: inherit; line-height: 1.4; min-height: 150px; overflow-y: auto; cursor: text; }
.colored-editor:focus-within { border-color: #9b7cb8; background: rgba(255,255,255,.95); }
.colored-editor .ce-line { padding: 5px 10px; margin: 2px 4px; border-radius: 4px; border-left: 4px solid; min-height: 1.4em; }
.colored-editor .ce-line:focus { outline: none; }
.colored-editor .ce-line.empty::before { content: attr(data-placeholder); color: #9a8baa; opacity: 0.6; }
.colored-editor .ce-line.empty:not(:first-child)::before { content: ''; }


.alternating-textarea { background: transparent; }

.syntax-editor { width: 100%; background: rgba(255,255,255,.8); border: 1px solid rgba(180,160,200,.3); border-radius: 10px; padding: 12px 16px; color: #4a3f5c; font-size: 14px; font-family: inherit; line-height: 1.6; min-height: 150px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; word-wrap: break-word; }
.syntax-editor:focus { outline: none; border-color: #9b7cb8; background: rgba(255,255,255,.95); }
.syntax-editor br { display: block; content: ''; margin: 0; }
.syntax-editor .hl-text { background: rgba(126,200,200,.3); color: #4a9090; border-radius: 3px; padding: 1px 2px; }
.syntax-editor .hl-question { background: rgba(196,144,209,.3); color: #8a5a9a; border-radius: 3px; padding: 1px 2px; }
.syntax-editor .hl-answer { background: rgba(229,192,123,.3); color: #a08050; border-radius: 3px; padding: 1px 2px; font-weight: 600; }
.text-lines-preview { margin-top: 10px; max-height: 200px; overflow-y: auto; border: 1px solid rgba(180,160,200,.2); border-radius: 8px; }
.text-line { padding: 8px 12px; font-size: 13px; border-bottom: 1px solid rgba(180,160,200,.1); color: #4a3f5c; }
.text-line:nth-child(odd) { background: rgba(196,144,209,.08); }
.text-line:nth-child(even) { background: rgba(126,200,200,.08); }
.text-line:last-child { border-bottom: none; }

#canvas { width: 100%; height: 100%; }

.floating-stats { position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); display: none; gap: 8px; z-index: 100; }
.floating-stats.visible { display: flex; }
.stat-card { background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.2); border-radius: 10px; padding: 6px 10px; text-align: center; backdrop-filter: blur(20px); min-width: 50px; box-shadow: 0 4px 15px rgba(100,80,120,.04); }
.stat-value { font-size: 12px; font-weight: 700; color: #9b7cb8; }
.stat-label { font-size: 7px; color: #9a8baa; text-transform: uppercase; letter-spacing: .08em; margin-top: 1px; }
.stat-errors { background: rgba(154,139,170,0.15); border-color: rgba(154,139,170,0.4); }
.stat-errors .stat-value { color: #9a8baa; }
.stat-errors .stat-label { color: #7a6b8a; }
.live-badge { display: flex; align-items: center; gap: 5px; background: rgba(255,255,255,.9); border: 1px solid rgba(196,144,209,.3); border-radius: 20px; padding: 6px 10px; }
.live-dot { width: 6px; height: 6px; background: #c490d1; border-radius: 50%; animation: pulse 1.5s ease infinite; box-shadow: 0 0 10px rgba(196,144,209,.5); }
.live-badge.paused .live-dot { animation: none; background: #b8b0c0; box-shadow: none; }
.live-text { font-size: 8px; font-weight: 600; color: #9b7cb8; text-transform: uppercase; }
.live-badge.paused .live-text { color: #9a8baa; }
@keyframes pulse { 0%,100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.5); opacity: .6; } }

.mode-toggle, .view-toggle { position: fixed; top: 20px; left: 20px; display: none; background: rgba(255,255,255,.95); border: 1px solid rgba(180,160,200,.25); border-radius: 30px; padding: 5px; z-index: 100; box-shadow: 0 8px 30px rgba(100,80,120,.05); }
.mode-toggle.visible, .view-toggle.visible { display: flex; }
.view-toggle { left: 20px; }
.evolution-mode-toggle { position: fixed; bottom: 25px; right: 20px; display: none; background: rgba(255,255,255,.95); border: 1px solid rgba(180,160,200,.25); border-radius: 30px; padding: 5px; z-index: 100; box-shadow: 0 8px 30px rgba(100,80,120,.05); }
.evolution-mode-toggle.visible { display: flex; }
.diversity-toggle { position: fixed; bottom: 25px; right: 280px; display: none; background: linear-gradient(135deg, rgba(255,240,245,.95), rgba(245,240,255,.95)); border-radius: 25px; padding: 10px 18px; z-index: 100; }
.diversity-toggle::before { content: ''; position: absolute; inset: 0; border-radius: 25px; padding: 2px; background: linear-gradient(90deg, #FF6B9D, #E5C07B, #98C379, #56B6C2, #C678DD); -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none; }
.diversity-toggle.visible { display: flex; align-items: center; }
.diversity-toggle .toggle-text { background: linear-gradient(90deg, #FF6B9D, #C678DD); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700; letter-spacing: .08em; }
.judge-model-toggle { position: fixed; top: 20px; right: 20px; display: none; background: rgba(255,255,255,.95); border: 1px solid rgba(180,160,200,.25); border-radius: 30px; padding: 5px; z-index: 100; box-shadow: 0 8px 30px rgba(100,80,120,.05); }
.judge-model-toggle.visible { display: flex; }
.toggle-label { display: flex; align-items: center; gap: 12px; cursor: pointer; user-select: none; }
.toggle-label input { display: none; }
.toggle-slider { width: 44px; height: 24px; background: rgba(180,160,200,.2); border-radius: 12px; position: relative; transition: all .3s; border: 1px solid rgba(180,160,200,.3); }
.toggle-slider::after { content: ''; position: absolute; width: 20px; height: 20px; background: white; border-radius: 50%; top: 1px; left: 1px; transition: all .3s; box-shadow: 0 2px 6px rgba(0,0,0,.15); }
.toggle-label input:checked + .toggle-slider { background: linear-gradient(90deg, #FF6B9D, #E5C07B, #98C379, #56B6C2, #C678DD); border-color: #C678DD; box-shadow: 0 0 15px rgba(198,120,221,.5); }
.toggle-label input:checked + .toggle-slider::after { left: 21px; box-shadow: 0 2px 8px rgba(0,0,0,.2); }
.toggle-text { font-size: 12px; font-weight: 700; color: #7a6b8a; text-transform: uppercase; letter-spacing: .08em; }
.mode-btn { padding: 10px 20px; border: none; background: transparent; color: #9a8baa; font-size: 12px; font-weight: 600; cursor: pointer; border-radius: 25px; transition: all .3s; }
.mode-btn.small { padding: 12px 20px; font-size: 13px; border-radius: 22px; }
.mode-btn.active { background: linear-gradient(135deg, #c490d1, #9b7cb8); color: white; }

/* Model checkbox list for multi-select */
.model-checkbox-list { display: flex; flex-wrap: wrap; gap: 8px; padding: 10px; background: rgba(250,248,252,.8); border: 1px solid rgba(180,160,200,.2); border-radius: 10px; min-height: 40px; max-height: 100px; overflow-y: auto; }
.model-checkbox-item { display: flex; align-items: center; gap: 6px; padding: 6px 12px; background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.25); border-radius: 20px; cursor: pointer; transition: all .2s; font-size: 12px; color: #6b5b7a; user-select: none; }
.model-checkbox-item:hover { border-color: #9b7cb8; background: white; }
.model-checkbox-item.selected { background: linear-gradient(135deg, #c490d1, #9b7cb8); border-color: #9b7cb8; color: white; box-shadow: 0 2px 8px rgba(155,124,184,.3); }
.model-checkbox-item input { display: none; }
.model-checkbox-item .check-icon { width: 14px; height: 14px; border: 2px solid rgba(180,160,200,.4); border-radius: 4px; display: flex; align-items: center; justify-content: center; transition: all .2s; font-size: 10px; }
.model-checkbox-item.selected .check-icon { background: white; border-color: white; color: #9b7cb8; }

.legend { position: fixed; bottom: 80px; left: 20px; display: none; flex-direction: column; gap: 6px; background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.2); border-radius: 12px; padding: 12px 16px; z-index: 100; max-width: 400px; max-height: 250px; overflow-y: auto; cursor: move; box-shadow: 0 8px 25px rgba(100,80,120,.04); }
.legend.visible { display: flex; }
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #4a3f5c; cursor: pointer; padding: 4px 8px; border-radius: 6px; transition: all .2s; }
.legend-item:hover { background: rgba(155,124,184,.1); }
.legend-item.dimmed { opacity: .25; }
.legend-swatch { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }

.control-buttons { position: fixed; bottom: 20px; left: 20px; display: none; gap: 10px; z-index: 100; }
.control-buttons.visible { display: flex; }
.control-btn { padding: 12px 24px; border: none; border-radius: 12px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all .3s; text-transform: uppercase; font-family: 'Roboto Mono', monospace; }
.control-btn.back { background: rgba(255,255,255,.8); border: 1px solid rgba(180,160,200,.3); color: #7a6b8a; }
.control-btn.back:hover { background: rgba(255,255,255,.95); border-color: #9b7cb8; color: #9b7cb8; }
.control-btn.stop { background: rgba(232,160,168,.2); border: 1px solid #e8a0a8; color: #c08088; }
.control-btn.stop:hover { background: #e8a0a8; color: white; }
.control-btn.reset { background: rgba(224,192,144,.2); border: 1px solid #e0c090; color: #b8a070; }
.control-btn.reset:hover { background: #e0c090; color: #4a3f5c; }
.control-btn.flip { background: rgba(126,200,200,.2); border: 1px solid #7ec8c8; color: #5a9898; }
.control-btn.flip:hover { background: #7ec8c8; color: #4a3f5c; }

.zoom-controls { position: fixed; bottom: 20px; right: 20px; display: none; flex-direction: column; gap: 2px; z-index: 100; }
.zoom-controls.visible { display: flex; }
.zoom-btn { width: 18px; height: 18px; border: 1px solid rgba(180,160,200,.3); border-radius: 4px; background: rgba(255,255,255,.9); color: #9a8baa; font-size: 10px; cursor: pointer; transition: all .2s; display: flex; align-items: center; justify-content: center; }
.zoom-btn:hover { border-color: #7ec8c8; color: #5a9898; background: rgba(126,200,200,.15); }


.trajectories-panel { position: fixed; top: 80px; right: 20px; width: 400px; max-height: calc(100vh - 120px); background: rgba(255,255,255,.95); border: 1px solid rgba(196,144,209,.3); border-radius: 16px; display: none; flex-direction: column; z-index: 200; box-shadow: 0 12px 50px rgba(100,80,120,.06); }
.trajectories-panel.visible { display: flex; }
.trajectories-header { display: flex; justify-content: space-between; align-items: center; padding: 15px 20px; border-bottom: 1px solid rgba(180,160,200,.2); cursor: move; }
.trajectories-title { font-size: 14px; font-weight: 600; color: #9b7cb8; font-family: 'Roboto Mono', monospace; }
.trajectories-close { background: none; border: none; color: #9a8baa; font-size: 22px; cursor: pointer; padding: 0 5px; transition: color .2s; }
.trajectories-close:hover { color: #e8a0a8; }
.trajectories-prefix { padding: 12px 20px; background: rgba(126,200,200,.1); font-size: 12px; color: #5a9898; border-bottom: 1px solid rgba(180,160,200,.15); font-style: italic; }
.trajectories-list { flex: 1; overflow-y: auto; padding: 10px; }
.trajectory-item { padding: 12px 15px; margin: 6px 0; background: rgba(248,244,255,.8); border-radius: 10px; border-left: 4px solid #9b7cb8; font-size: 12px; line-height: 1.5; color: #4a3f5c; transition: all .2s; }
.trajectory-item:hover { background: rgba(248,244,255,1); }
.trajectory-prefill { color: #9a8baa; font-style: italic; }
.trajectory-generated { color: #4a3f5c; }
.trajectory-scores { margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(180,160,200,.2); display: flex; gap: 8px; flex-wrap: wrap; }
.trajectory-score { font-size: 10px; padding: 2px 6px; border-radius: 4px; color: white; opacity: 0.9; }

.tooltip { position: fixed; background: rgba(255,255,255,.98); border: 1px solid rgba(196,144,209,.3); border-radius: 12px; padding: 15px 20px; pointer-events: none; opacity: 0; transition: opacity .2s; z-index: 1000; min-width: 220px; max-width: 400px; box-shadow: 0 12px 40px rgba(100,80,120,.06); }
.tooltip.visible { opacity: 1; }
.tooltip-title { font-size: 14px; font-weight: 700; color: #9b7cb8; margin-bottom: 10px; font-family: 'Roboto Mono', monospace; }
.tooltip-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; color: #7a6b8a; }
.tooltip-value { font-weight: 600; color: #5a9898; }
.tooltip-raw { margin-top: 10px; padding: 10px; background: rgba(248,244,255,.8); border-radius: 6px; font-size: 11px; color: #6a5a7a; font-family: monospace; white-space: pre-wrap; word-break: break-word; }

.node-hit-area { fill: transparent; cursor: pointer; }
.node-card { fill: rgba(255,255,255,0.9); stroke: rgba(180,160,200,.3); stroke-width: 1px; transition: all .3s; cursor: pointer; pointer-events: none; }
.node:hover .node-card { stroke: #c490d1; stroke-width: 2px; filter: drop-shadow(0 0 25px rgba(196,144,209,.25)); }
.node-card.reference { stroke: #7ec8c8; stroke-width: 2px; filter: drop-shadow(0 0 20px rgba(126,200,200,.3)); }
.node-card.greyed { opacity: .15; filter: grayscale(100%) brightness(0.8); }
.node-label { font-size: 11px; font-weight: 600; fill: #4a3f5c; }
.node-info { font-size: 9px; fill: #9a8baa; }
.link { fill: none; stroke-width: 2.5px; stroke-linecap: round; }
.bar { transition: all .4s; cursor: pointer; }
.bar:hover { filter: brightness(1.15); }

.node-text-group { transition: opacity 0.3s; }
.node-unique-text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.node { cursor: pointer; transition: filter 0.2s, opacity 0.2s; }
.node:hover .node-unique-text { fill: #9b7cb8 !important; }
.node:hover .node-card { stroke: #c490d1; }

/* Orientation mode - faded for ancestors (CSS-driven for performance) */
/* Smooth transitions for all orientation changes - explicit defaults for reset */
.node { transition: filter 0.35s ease-out, opacity 0.35s ease-out; filter: none; opacity: 1; }
.node.orient-ancestor { filter: saturate(0.3) brightness(1.1) contrast(0.9); opacity: 0.4; }
.node.orient-ancestor .node-text-group { opacity: 0.3; }
.node .node-text-group { transition: opacity 0.35s ease-out; opacity: 1; }
.link { transition: stroke 0.35s ease-out, opacity 0.35s ease-out, stroke-width 0.35s ease-out; }
.link.orient-ancestor { stroke: rgba(180,160,200,0.25) !important; opacity: 0.35 !important; }
.field-glow { transition: opacity 0.35s ease-out; opacity: 1; }
.field-glow.orient-ancestor { opacity: 0.2 !important; }
.phantom-canvas-fade { transition: filter 0.4s ease-out, opacity 0.4s ease-out; filter: none; opacity: 1; }
.phantom-canvas-fade.orient-ancestor { filter: saturate(0.3) brightness(1.2) contrast(0.8); opacity: 0.2; }
.prompt-box-group { transition: filter 0.35s ease-out, opacity 0.35s ease-out; filter: none; opacity: 1; }
.prompt-box-group.orient-ancestor { filter: saturate(0.3) brightness(1.1) contrast(0.85); opacity: 0.3; }
.bar { transition: all 0.3s ease-out; }

.chart-line { fill: none; stroke-width: 2.5px; }
.chart-dot { cursor: pointer; transition: all .2s; }
.chart-dot:hover { r: 8; }
.chart-area { opacity: .15; }
.axis text { fill: #9a8baa; font-size: 11px; }
.axis line, .axis path { stroke: rgba(180,160,200,.3); }
.axis.diversity-axis text { fill: #C678DD; font-weight: 700; }
.axis.diversity-axis line, .axis.diversity-axis path { stroke: #C678DD; stroke-width: 2px; }
.diversity-line { filter: drop-shadow(0 0 6px rgba(198,120,221,.6)); }
.diversity-dot { filter: drop-shadow(0 0 8px rgba(198,120,221,.7)); }
.grid line { stroke: rgba(180,160,200,.15); }

.text-display { position: fixed; bottom: 80px; right: 20px; background: rgba(255,255,255,.95); border: 1px solid rgba(180,160,200,.2); border-radius: 12px; padding: 15px 20px; display: none; z-index: 100; max-width: 400px; max-height: 150px; overflow-y: auto; box-shadow: 0 8px 25px rgba(100,80,120,.04); }
.text-display.visible { display: block; }
.text-display-label { font-size: 10px; color: #9a8baa; text-transform: uppercase; margin-bottom: 8px; }
.text-display-content { font-size: 13px; line-height: 1.5; color: #4a3f5c; }
.text-highlight { background: rgba(196,144,209,.3); border-radius: 2px; }

.pie-chart-panel { position: fixed; top: 20px; left: 20px; background: rgba(255,255,255,.95); border: 1px solid rgba(180,160,200,.2); border-radius: 16px; padding: 15px; display: none; z-index: 100; cursor: move; box-shadow: 0 8px 25px rgba(100,80,120,.04); }
.pie-chart-panel.visible { display: block; }

.add-texts-panel { position: fixed; bottom: 20px; right: 20px; background: rgba(255,255,255,.95); border: 1px solid rgba(126,200,200,.3); border-radius: 16px; padding: 20px; display: none; z-index: 100; width: 420px; box-shadow: 0 12px 50px rgba(100,80,120,.05); cursor: move; }
.add-texts-panel .config-textarea, .add-texts-panel .colored-editor, .add-texts-panel button { cursor: auto; }
.add-texts-panel-header { cursor: move; margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid rgba(180,160,200,.2); }
.add-texts-panel.visible { display: block; }
.add-texts-panel .config-label { margin-bottom: 10px; }
.add-texts-btn { width: 100%; padding: 12px; font-size: 13px; margin-top: 10px; }

.progress-bar { position: fixed; top: 0; left: 0; height: 3px; background: linear-gradient(90deg, #c490d1, #9b7cb8, #7ec8c8); transition: width .3s; z-index: 2000; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(248,244,255,.5); }
::-webkit-scrollbar-thumb { background: rgba(196,144,209,.4); border-radius: 4px; }

.toast-container { position: fixed; top: 80px; right: 20px; z-index: 3000; display: flex; flex-direction: column; gap: 10px; max-width: 400px; pointer-events: none; }
.toast { background: rgba(255,255,255,.98); border-radius: 12px; padding: 14px 18px; box-shadow: 0 10px 40px rgba(100,80,120,.06); border-left: 4px solid; animation: toastSlideIn 0.3s ease; pointer-events: auto; display: flex; align-items: flex-start; gap: 12px; }
.toast.error { border-left-color: #e8a0a8; }
.toast.warning { border-left-color: #e0c090; }
.toast.info { border-left-color: #7ec8c8; }
.toast.success { border-left-color: #98c8a0; }
.toast-icon { font-size: 18px; flex-shrink: 0; }
.toast.error .toast-icon { color: #c08088; }
.toast.warning .toast-icon { color: #b8a070; }
.toast.info .toast-icon { color: #5a9898; }
.toast.success .toast-icon { color: #689870; }
.toast-content { flex: 1; }
.toast-title { font-size: 13px; font-weight: 600; color: #4a3f5c; margin-bottom: 3px; }
.toast-message { font-size: 11px; color: #7a6b8a; line-height: 1.4; word-break: break-word; }
.toast-close { background: none; border: none; color: #9a8baa; font-size: 18px; cursor: pointer; padding: 0; margin-left: 8px; opacity: 0.6; transition: opacity 0.2s; }
.toast-close:hover { opacity: 1; }
.toast.hiding { animation: toastSlideOut 0.3s ease forwards; }
@keyframes toastSlideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
@keyframes toastSlideOut { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }

/* Trajectory Explorer Panel */
.traj-explorer { position: fixed; inset: 0; background: linear-gradient(135deg, #fef6f9 0%, #f8f4ff 30%, #f0f7ff 60%, #fff5f0 100%); z-index: 2500; display: none; flex-direction: column; padding: 60px 40px 40px; }
.traj-explorer.visible { display: flex; }
.traj-explorer-close { position: absolute; top: 20px; right: 25px; background: none; border: none; color: #9a8baa; font-size: 32px; cursor: pointer; transition: color .2s; z-index: 10; }
.traj-explorer-close:hover { color: #e8a0a8; }
.traj-explorer-hint { position: absolute; top: 22px; left: 50%; transform: translateX(-50%); font-size: 11px; color: #9a8baa; letter-spacing: .05em; opacity: .7; }
.traj-explorer-content { display: flex; gap: 40px; flex: 1; max-width: 1600px; margin: 0 auto; width: 100%; }
.traj-explorer-text-section { flex: 1.5; display: flex; flex-direction: column; gap: 20px; }
.traj-explorer-chart-section { flex: 1; display: flex; flex-direction: column; gap: 15px; min-width: 350px; max-width: 450px; }

.traj-explorer-prompt { background: linear-gradient(135deg, rgba(126,200,200,.15), rgba(100,180,180,.08)); border: 2px solid rgba(126,200,200,.4); border-radius: 14px; padding: 18px 24px; font-size: 15px; color: #3a7878; font-weight: 500; line-height: 1.7; box-shadow: 0 3px 15px rgba(100,180,180,.12); flex-shrink: 0; }
.traj-explorer-text { background: rgba(255,255,255,.9); border: 1px solid rgba(196,144,209,.2); border-radius: 16px; padding: 35px 40px; font-size: 20px; line-height: 1.9; flex: 1; overflow-y: auto; box-shadow: 0 4px 20px rgba(100,80,120,.08); }
.traj-explorer-text .word { display: inline; cursor: pointer; padding: 2px 1px; border-radius: 3px; transition: all .15s; }
.traj-explorer-text .word:hover { background: rgba(196,144,209,.15); }
.traj-explorer-text .word.past { color: #4a3f5c; opacity: 1; }
.traj-explorer-text .word.current { background: linear-gradient(135deg, rgba(196,144,209,.4), rgba(155,124,184,.4)); color: #4a3f5c; font-weight: 600; padding: 3px 6px; }
.traj-explorer-text .word.future { color: #9a8baa; opacity: .5; }

.traj-explorer-slider-track { height: 40px; background: rgba(255,255,255,.8); border-radius: 20px; position: relative; cursor: pointer; border: 1px solid rgba(180,160,200,.25); user-select: none; -webkit-user-select: none; }
.traj-explorer.dragging { cursor: grabbing; }
.traj-explorer.dragging * { user-select: none; -webkit-user-select: none; }
.traj-explorer-slider-ball { position: absolute; width: 32px; height: 32px; top: 3px; border-radius: 50%; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.9), #c490d1); box-shadow: 0 0 25px rgba(196,144,209,.5), 0 0 50px rgba(196,144,209,.3), inset 0 0 15px rgba(255,255,255,.3); transition: left .1s ease-out, background .3s, box-shadow .3s; pointer-events: none; }
.traj-explorer-slider-ball::after { content: ''; position: absolute; inset: 3px; border-radius: 50%; background: radial-gradient(circle at 35% 35%, rgba(255,255,255,.6), transparent 60%); }
.traj-explorer-diversity { font-size: 12px; color: #9a8baa; text-align: center; padding: 8px 0; }
.traj-explorer-diversity span { background: linear-gradient(90deg, #FF6B9D, #C678DD); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700; font-size: 14px; }

.traj-explorer-mode-toggle { display: flex; background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.2); border-radius: 25px; padding: 4px; }
.traj-mode-btn { flex: 1; padding: 10px 15px; border: none; background: transparent; color: #9a8baa; font-size: 11px; font-weight: 600; cursor: pointer; border-radius: 20px; transition: all .3s; text-transform: uppercase; letter-spacing: .05em; }
.traj-mode-btn.active { background: linear-gradient(135deg, #c490d1, #9b7cb8); color: white; }
.traj-mode-btn:hover:not(.active) { color: #7a6b8a; }

.traj-explorer-chart { width: 100%; flex: 1; min-height: 300px; background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.15); border-radius: 14px; }
.traj-explorer-chart .bar-bg { fill: rgba(248,244,255,.8); }
.traj-explorer-chart .bar-fill { transition: width .15s ease-out; }
.traj-explorer-chart .bar-label { fill: #4a3f5c; font-size: 11px; font-weight: 500; }
.traj-explorer-chart .bar-value { fill: #9a8baa; font-size: 10px; font-family: monospace; }
.traj-explorer-chart .chart-title { fill: #9a8baa; font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }

.traj-explorer-legend { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; padding: 12px 16px; background: rgba(255,255,255,.9); border: 1px solid rgba(180,160,200,.15); border-radius: 12px; }
.traj-explorer-legend .legend-item { display: flex; align-items: center; gap: 6px; font-size: 10px; color: #4a3f5c; }
.traj-explorer-legend .legend-swatch { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }

/* Convergence Panel */
.convergence-panel { position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%); background: rgba(255,255,255,.95); border: 1px solid rgba(180,160,200,.25); border-radius: 16px; padding: 15px 25px; display: none; z-index: 100; width: 600px; max-width: 90vw; box-shadow: 0 8px 30px rgba(100,80,120,.08); }
.convergence-panel.visible { display: block; }
.convergence-position-slider { display: flex; flex-direction: column; gap: 10px; }
.convergence-slider-label { font-size: 12px; color: #7a6b8a; text-align: center; font-weight: 600; }
.convergence-slider-label span { color: #9b7cb8; font-weight: 700; }
.convergence-slider-track { height: 32px; background: linear-gradient(90deg, rgba(126,200,200,.15), rgba(196,144,209,.15)); border-radius: 16px; position: relative; cursor: pointer; border: 1px solid rgba(180,160,200,.3); user-select: none; }
.convergence-slider-ball { position: absolute; width: 26px; height: 26px; top: 2px; left: 2px; border-radius: 50%; background: radial-gradient(circle at 30% 30%, rgba(255,255,255,.9), #7ec8c8); box-shadow: 0 0 15px rgba(126,200,200,.5), inset 0 0 10px rgba(255,255,255,.3); transition: left .1s ease-out; pointer-events: none; }
"""
