"""Shared app JavaScript: state, config, websocket, mode selection."""

from __future__ import annotations


def get_shared_app_js() -> str:
    return """
// ════════════════════════════════════════════════════════════════════════════════
// APPLICATION STATE
// ════════════════════════════════════════════════════════════════════════════════
let ws=null, currentMode=null;
let treeState={nodes:[],questions:[]}, dynamicsState={positions:[],questions:[],continuation:'',prefill:'',generated:''}, judgeState={texts:[],questions:[],results:[]};
let judgeAddingMore=false;  // Flag to prevent results reset when adding texts
let showOrientation=false, showMagnitudes=false, referenceNode='root', highlightedQuestion=null, highlightedMagnitude=null;
let appConfig={anthropic_key:'',openai_key:'',models:{},provider_names:{},defaults:{}};
let settings={gen_provider:'openai',gen_model:'gpt-4o-mini',judge_model:[{provider:'openai',model:'gpt-4o-mini'}],gen_temperature:1.0,judge_temperature:0.0,max_tokens:300,judge_prompt:''};
const colors=['#FF6B9D','#C678DD','#56B6C2','#E5C07B','#98C379','#61AFEF','#E06C75','#D19A66'];

// ════════════════════════════════════════════════════════════════════════════════
// CONFIG & SETTINGS
// ════════════════════════════════════════════════════════════════════════════════
async function loadConfig(){
    try{const r=await fetch('/config');appConfig=await r.json();
    settings={...appConfig.defaults};

    // Auto-select provider based on available API keys
    const autoProvider = autoSelectProvider();
    if (autoProvider) {
        settings.gen_provider = autoProvider.provider;
        settings.gen_model = autoProvider.model;
        settings.judge_model = [{provider: autoProvider.provider, model: autoProvider.model}];
        console.log('🔑 Auto-selected provider:', autoProvider.provider, 'model:', autoProvider.model);
    }

    // Populate generation model list (single-select)
    populateSingleSelectModelList('settingsGenModelList', {provider: settings.gen_provider, model: settings.gen_model});
    // Populate judge model list (multi-select)
    populateMultiProviderModelList('settingsJudgeModelList', settings.judge_model);
    document.getElementById('settingsGenTemp').value=settings.gen_temperature||1.0;
    document.getElementById('genTempVal').textContent=settings.gen_temperature||1.0;
    document.getElementById('settingsJudgeTemp').value=settings.judge_temperature||0.0;
    document.getElementById('judgeTempVal').textContent=settings.judge_temperature||0.0;
    document.getElementById('settingsMaxTokens').value=settings.max_tokens||300;
    document.getElementById('maxTokensVal').textContent=settings.max_tokens||300;
    initSyntaxEditor('settingsJudgePrompt',settings.judge_prompt);
    document.getElementById('settingsAnthropicKey').value=appConfig.anthropic_key;
    document.getElementById('settingsOpenaiKey').value=appConfig.openai_key;
    initSyntaxEditor('judgePromptFormat',settings.judge_prompt);
    // Init judge config screen from global settings
    populateMultiProviderModelList('judgeModelList', settings.judge_model);
    // Init temperature slider in judge config
    const judgeTempConfig = document.getElementById('judgeTemperatureConfig');
    if(judgeTempConfig) {
        judgeTempConfig.value = settings.judge_temperature || 0.0;
        document.getElementById('judgeTempValConfig').textContent = settings.judge_temperature || 0.0;
    }

    // Show toast if no API keys found and falling back to local
    if (autoProvider && isLocalProvider(autoProvider.provider)) {
        showToast('info', 'Using Local Models', 'No API keys found. Using HuggingFace models (requires local GPU).', 5000);
    } else if (!autoProvider) {
        showToast('warning', 'No API Keys', 'Configure API keys in Settings or use HuggingFace local models.', 8000);
    }
    }catch(e){console.error(e)}}

// Auto-select provider based on available API keys
// Priority: OpenAI (smallest) > Anthropic (haiku) > HuggingFace Instruct (local)
function autoSelectProvider() {
    const openaiKey = appConfig.openai_key?.trim();
    const anthropicKey = appConfig.anthropic_key?.trim();

    if (openaiKey && openaiKey.length > 0) {
        const models = appConfig.models?.openai || [];
        const smallestModel = models.find(m => m.includes('mini')) || models[0] || 'gpt-4o-mini';
        return { provider: 'openai', model: smallestModel };
    }

    if (anthropicKey && anthropicKey.length > 0) {
        const models = appConfig.models?.anthropic || [];
        const haikuModel = models.find(m => m.includes('haiku')) || models[0] || 'claude-haiku-4-5';
        return { provider: 'anthropic', model: haikuModel };
    }

    // No API keys - fall back to local HuggingFace Instruct
    const hfModels = appConfig.models?.huggingface_instruct || [];
    const smallestModel = hfModels.find(m => m.includes('0.8B')) || hfModels[0];
    if (smallestModel) {
        return { provider: 'huggingface_instruct', model: smallestModel };
    }

    return null;
}

// Populate single-select model list (for generation - only one can be selected)
function populateSingleSelectModelList(containerId, selectedModel=null){
    const container=document.getElementById(containerId);
    if(!container)return;
    container.innerHTML='';

    const allProviders=getAllProviders();
    const providerColors={openai:'#10a37f',anthropic:'#cc785c',huggingface_base:'#ffd21e',huggingface_instruct:'#ffd21e',huggingface_reasoning:'#ffd21e'};

    // Convert selected to key format
    const selectedKey=selectedModel?`${selectedModel.provider}/${selectedModel.model}`:null;

    allProviders.forEach(provider=>{
        const models=appConfig.models[provider]||[];
        if(!models.length)return;

        // Provider header
        const header=document.createElement('div');
        header.className='model-provider-header';
        const color=providerColors[provider]||'#888';
        header.style.cssText=`width:100%;font-size:10px;font-weight:600;color:${color};margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;`;
        header.textContent=appConfig.provider_names[provider]||provider;
        container.appendChild(header);

        // Models for this provider
        models.forEach(model=>{
            const modelKey=`${provider}/${model}`;
            const isSelected=modelKey===selectedKey;
            const item=document.createElement('label');
            item.className='model-checkbox-item'+(isSelected?' selected':'');
            item.dataset.provider=provider;
            item.dataset.model=model;
            item.innerHTML=`<input type="radio" name="${containerId}" value="${modelKey}" ${isSelected?'checked':''}><span class="check-icon">\u2713</span><span>${model}</span>`;
            item.querySelector('input').addEventListener('change',function(){
                // Deselect all others
                container.querySelectorAll('.model-checkbox-item').forEach(i=>i.classList.remove('selected'));
                item.classList.add('selected');
            });
            container.appendChild(item);
        });
    });
}

// Get single selected model as {provider, model}
function getSelectedModelFromList(containerId){
    const container=document.getElementById(containerId);
    if(!container)return null;
    const checked=container.querySelector('input:checked');
    if(!checked)return null;
    const item=checked.closest('.model-checkbox-item');
    return {provider:item.dataset.provider,model:item.dataset.model};
}

// Get all providers from config
function getAllProviders(){
    return Object.keys(appConfig.models||{});
}

// Check if provider is local (doesn't need API key)
function isLocalProvider(provider){
    return provider.startsWith('huggingface_');
}

// Populate multi-provider judge model list (shows all providers)
function populateMultiProviderModelList(containerId, selectedModels=[]){
    const container=document.getElementById(containerId);
    if(!container)return;
    container.innerHTML='';

    const allProviders=getAllProviders();
    const providerColors={openai:'#10a37f',anthropic:'#cc785c',huggingface_base:'#ffd21e',huggingface_instruct:'#ffd21e',huggingface_reasoning:'#ffd21e'};

    // Convert selectedModels to Set of "provider/model" strings for easy lookup
    const selectedSet=new Set(selectedModels.map(m=>typeof m==='object'?`${m.provider}/${m.model}`:m));

    allProviders.forEach(provider=>{
        const models=appConfig.models[provider]||[];
        if(!models.length)return;

        // Provider header with display name
        const header=document.createElement('div');
        header.className='model-provider-header';
        const color=providerColors[provider]||'#888';
        header.style.cssText=`width:100%;font-size:10px;font-weight:600;color:${color};margin-top:8px;margin-bottom:4px;text-transform:uppercase;letter-spacing:0.05em;`;
        header.textContent=appConfig.provider_names[provider]||provider;
        container.appendChild(header);

        // Models for this provider
        models.forEach(model=>{
            const modelKey=`${provider}/${model}`;
            const isSelected=selectedSet.has(modelKey);
            const item=document.createElement('label');
            item.className='model-checkbox-item'+(isSelected?' selected':'');
            item.dataset.provider=provider;
            item.dataset.model=model;
            item.innerHTML=`<input type="checkbox" value="${modelKey}" ${isSelected?'checked':''}><span class="check-icon">\u2713</span><span>${model}</span>`;
            item.querySelector('input').addEventListener('change',function(){
                item.classList.toggle('selected',this.checked);
            });
            container.appendChild(item);
        });
    });
}

// Get selected models as [{provider, model}] format
function getSelectedModelsFromList(containerId){
    const container=document.getElementById(containerId);
    if(!container)return[];
    return Array.from(container.querySelectorAll('input:checked')).map(cb=>{
        const item=cb.closest('.model-checkbox-item');
        return {provider:item.dataset.provider,model:item.dataset.model};
    });
}

function updateJudgeModels(){
    // Refresh judge model list with all available providers
    populateMultiProviderModelList('settingsJudgeModelList', settings.judge_model);
}

function updateJudgeConfigModels(){
    // Refresh judge model list with all available providers
    populateMultiProviderModelList('judgeModelList', settings.judge_model);
}
function showSettings(){
    console.log('showSettings: opening settings modal');
    document.getElementById('settingsModal').classList.add('visible');
    console.log('showSettings: landing hidden?', document.getElementById('landing').classList.contains('hidden'));
}
function hideSettings(){
    console.log('hideSettings: closing settings modal');
    document.getElementById('settingsModal').classList.remove('visible');
    console.log('hideSettings: landing hidden?', document.getElementById('landing').classList.contains('hidden'));
    // Make sure landing is visible if no mode is selected
    if(!currentMode){
        console.log('hideSettings: no currentMode, ensuring landing is visible');
        document.getElementById('landing').classList.remove('hidden');
    }
}
function saveSettings(){
    // Get selected generation model (single-select)
    const genModel=getSelectedModelFromList('settingsGenModelList');
    if(genModel){
        settings.gen_provider=genModel.provider;
        settings.gen_model=genModel.model;
    }
    // Get selected judge models (multi-select)
    settings.judge_model=getSelectedModelsFromList('settingsJudgeModelList');
    settings.gen_temperature=parseFloat(document.getElementById('settingsGenTemp').value);
    settings.judge_temperature=parseFloat(document.getElementById('settingsJudgeTemp').value);
    settings.max_tokens=parseInt(document.getElementById('settingsMaxTokens').value);
    settings.judge_prompt=getSyntaxEditorValue('settingsJudgePrompt');
    appConfig.anthropic_key=document.getElementById('settingsAnthropicKey').value;
    appConfig.openai_key=document.getElementById('settingsOpenaiKey').value;
    initSyntaxEditor('judgePromptFormat',settings.judge_prompt);
    // Sync judge config screen to match Settings modal
    populateMultiProviderModelList('judgeModelList',settings.judge_model);
    const judgeTempConfig=document.getElementById('judgeTemperatureConfig');
    if(judgeTempConfig){
        judgeTempConfig.value=settings.judge_temperature;
        document.getElementById('judgeTempValConfig').textContent=settings.judge_temperature;
    }
    hideSettings();
    resetExperiment();
}
function resetSettings(){loadConfig();hideSettings()}
function getApiKeys(){return {openai: appConfig.openai_key, anthropic: appConfig.anthropic_key, huggingface_base: '', huggingface_instruct: '', huggingface_reasoning: ''}}

// ════════════════════════════════════════════════════════════════════════════════
// WEBSOCKET
// ════════════════════════════════════════════════════════════════════════════════
function connect(){const p=location.protocol==='https:'?'wss:':'ws:';ws=new WebSocket(`${p}//${location.host}/ws`);ws.onopen=()=>console.log('Connected');ws.onclose=()=>setTimeout(connect,1000);ws.onmessage=handleMessage}

function handleMessage(e){
    try {
        const m=JSON.parse(e.data);
        console.log('📨 WS message received:', m.type, m);

        if(m.type==='error'){
            console.error('❌ Error:', m.message || m.data?.message);
            const errorData = m.data || {message: m.message};
            const title = errorData.error_type || 'Error';
            const msg = errorData.message || 'An unknown error occurred';
            const nodeInfo = errorData.node_name ? ` (node: ${errorData.node_name})` : '';
            showToast('error', title + nodeInfo, msg);
            return;
        }
        if(m.type==='started'){
            console.log('🚀 Started event, mode=', m.mode, 'nodes=', m.data?.nodes?.length);
            showToast('info', 'Started', `Initializing ${m.mode} with ${m.data?.nodes?.length || 0} nodes`, 3000);
            if(m.mode==='tree'){
                treeState=m.data;
                console.log('🌳 Tree state initialized with', treeState.nodes?.length, 'nodes:', treeState.nodes?.map(n => n.node_id));
                try {
                    initTreeViz();
                    console.log('🌳 initTreeViz completed successfully');
                    showToast('success', 'Tree Ready', `Visualization initialized with ${treeState.nodes?.length} nodes`, 2000);
                } catch(vizErr) {
                    console.error('💥 initTreeViz FAILED:', vizErr);
                    showToast('error', 'Visualization Error', vizErr.message);
                }
            }
            else if(m.mode==='dynamics'){
                // Preserve prompt and continuation if already set
                const prevPrompt = dynamicsState.prompt;
                const prevContinuation = dynamicsState.continuation;
                dynamicsState = {...m.data, prompt: prevPrompt || m.data.prompt, continuation: prevContinuation || m.data.continuation};
                console.log('📊 Dynamics started, prompt:', dynamicsState.prompt?.length, 'chars, continuation:', dynamicsState.continuation?.length, 'chars');
                initDynamicsViz();
            }
            else if(m.mode==='judge'){
                    if(judgeAddingMore){
                        // Adding more texts - preserve existing results and texts
                        judgeState.questions=m.data.questions||judgeState.questions;
                        judgeAddingMore=false;
                    }else{
                        // Fresh start - reset results
                        judgeState={...judgeState,...m.data,results:[]};
                        initJudgeViz();
                    }
                }
        }
        else if(m.type==='node_update'){
            console.log('📊 node_update for node_id=', m.data?.node_id, 'n_samples=', m.data?.n_samples);
            console.log('📊 nodeGroups defined?', typeof nodeGroups !== 'undefined' && nodeGroups !== null);
            console.log('📊 treeState.nodes:', treeState?.nodes?.length);
            try {
                updateTreeNode(m.data);
                console.log('📊 updateTreeNode completed');
            } catch(updateErr) {
                console.error('💥 updateTreeNode FAILED:', updateErr);
                showToast('error', 'Update Failed', `Node ${m.data?.node_id}: ${updateErr.message}`, 5000);
            }
        }
        else if(m.type==='position_update')updateDynamicsPosition(m.data);
        else if(m.type==='text_scored')updateJudgeResult(m.data);
        else if(m.type==='continuation'){
            console.log('📊 Continuation received:', m.text?.length, 'chars, prefill:', m.prefill?.length, 'generated:', m.generated?.length);
            dynamicsState.continuation=m.text;
            dynamicsState.prefill=m.prefill||'';
            dynamicsState.generated=m.generated||'';
            // Show with prefill dimmed, generated normal
            const tc = document.getElementById('textContent');
            tc.innerHTML = '';
            if (m.prefill) {
                const pf = document.createElement('span');
                pf.className = 'trajectory-prefill';
                pf.textContent = m.prefill;
                tc.appendChild(pf);
            }
            if (m.generated) {
                const gen = document.createElement('span');
                gen.className = 'trajectory-generated';
                gen.textContent = m.generated;
                tc.appendChild(gen);
            }
        }
        else if(m.type==='complete'){
            console.log('✅ Complete event');
            document.getElementById('liveBadge').classList.add('paused');
            document.getElementById('liveText').textContent='Complete';
            document.getElementById('progressBar').style.width='100%';
        }
        else if(m.type==='status')document.getElementById('liveText').textContent=m.message.slice(0,20)
        else console.warn('⚠️ Unknown message type:', m.type);
    } catch(err) {
        console.error('💥 handleMessage EXCEPTION:', err);
        console.error('💥 Raw message data:', e.data);
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// MODE SELECTION & NAVIGATION
// ════════════════════════════════════════════════════════════════════════════════
function selectMode(m){document.getElementById('landing').classList.add('hidden');document.getElementById(m+'Config').classList.add('visible')}
function showLanding(){['treeConfig','dynamicsConfig','judgeConfig'].forEach(id=>document.getElementById(id).classList.remove('visible'));document.getElementById('landing').classList.remove('hidden')}
function goBack(){
    // Clear mode first so resetExperiment shows landing instead of config
    currentMode=null;
    resetExperiment();
}
function resetExperiment(){
    console.log('resetExperiment: currentMode=', currentMode);
    // In judge mode, just clear results instead of full reset
    if(currentMode==='judge'){
        resetJudgeResults();
        return;
    }
    stopSampling();
    ['floatingStats','modeToggle','viewToggle','evolutionModeToggle','diversityToggle','judgeModelToggle','controlButtons','legend','textDisplay','addTextsPanel','trajectoriesPanel','pieChartPanel','zoomControls','convergencePanel','trajExplorer'].forEach(id=>{
        const el=document.getElementById(id);
        if(el)el.classList.remove('visible');
    });
    document.getElementById('progressBar').style.width='0%';
    d3.select('#canvas').selectAll('*').remove();
    treeState={nodes:[],questions:[]};
    dynamicsState={positions:[],questions:[],continuation:'',prefill:'',generated:''};
    judgeState={texts:[],questions:[],results:[]};
    judgeAddingMore=false;  // Reset flag on experiment reset
    // Reset judge-specific state (defined in judge_page_js)
    if(typeof resetJudgeState === 'function') resetJudgeState();
    document.getElementById('liveBadge').classList.remove('paused');
    document.getElementById('liveText').textContent='Sampling';
    // Reset diversity toggle state
    const diversityCheckbox = document.getElementById('diversityCheckbox');
    if(diversityCheckbox) diversityCheckbox.checked = false;
    if(currentMode){
        document.getElementById(currentMode+'Config').classList.add('visible');
    } else {
        // No mode selected, show landing
        document.getElementById('landing').classList.remove('hidden');
    }
}

function resetJudgeResults(){
    // Just clear judge results without leaving the UI or changing model selection
    stopSampling();
    judgeState.texts=[];
    judgeState.results=[];
    judgeAddingMore=false;
    // Only reset results-related state, NOT model selection (judgeModels preserved)
    if(typeof resultsByModel !== 'undefined') resultsByModel={};
    if(typeof currentModelView !== 'undefined') currentModelView='averaged';
    document.getElementById('progressBar').style.width='0%';
    document.getElementById('liveBadge').classList.remove('paused');
    document.getElementById('liveText').textContent='Ready';
    // Redraw empty heatmap and pie
    d3.select('#canvas').selectAll('*').remove();
    if(typeof drawPieChart === 'function') drawPieChart();
    // Update model toggle (will show based on judgeModels which is preserved)
    if(typeof updateModelToggle === 'function') updateModelToggle();
}
function stopSampling(){if(ws&&ws.readyState===1)ws.send(JSON.stringify({action:'stop'}));document.getElementById('liveBadge').classList.add('paused');document.getElementById('liveText').textContent='Stopped'}
function hideTooltip(){document.getElementById('tooltip').classList.remove('visible')}

// ════════════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ════════════════════════════════════════════════════════════════════════════════
function showToast(type, title, message, duration=8000) {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = {error: '⚠️', warning: '⚡', info: 'ℹ️', success: '✓'};
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || '•'}</span>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">×</button>
    `;
    container.appendChild(toast);
    // Auto-dismiss after duration
    if (duration > 0) {
        setTimeout(() => {
            if (toast.parentElement) {
                toast.classList.add('hiding');
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    }
    // Keep max 5 toasts
    while (container.children.length > 5) {
        container.firstChild.remove();
    }
    console.log(`🔔 Toast [${type}]: ${title} - ${message}`);
}

// ════════════════════════════════════════════════════════════════════════════════
// COLORED EDITOR - Questions with legend colors
// ════════════════════════════════════════════════════════════════════════════════
function initColoredEditor(id, initialText, placeholder, colorOffset) {
    const editor = document.getElementById(id);
    if (!editor) return;
    editor.innerHTML = '';
    editor._colorOffset = colorOffset || 0;
    editor._placeholder = placeholder || '';
    const lines = initialText ? initialText.split('\\n') : [''];
    lines.forEach((text, i) => createColoredLine(editor, text, i, placeholder, editor._colorOffset));
    editor.addEventListener('keydown', handleColoredEditorKey);
    editor.addEventListener('input', recolorAllLines);
    editor.addEventListener('paste', handleColoredPaste);
    editor.addEventListener('click', handleColoredEditorClick);
}

function handleColoredEditorClick(e) {
    const editor = e.currentTarget;
    // If clicked on editor background (not on a line), focus last line
    if (e.target === editor) {
        const lastLine = editor.querySelector('.ce-line:last-child');
        if (lastLine) {
            lastLine.focus();
            // Put cursor at end
            const range = document.createRange();
            range.selectNodeContents(lastLine);
            range.collapse(false);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
        }
    }
}

function createColoredLine(editor, text, index, placeholder, colorOffset) {
    const line = document.createElement('div');
    line.className = 'ce-line' + (text.trim() === '' ? ' empty' : '');
    line.contentEditable = 'true';
    line.textContent = text;
    const offset = colorOffset || editor._colorOffset || 0;
    line.style.borderLeftColor = colors[(index + offset) % colors.length];
    if (placeholder) line.setAttribute('data-placeholder', placeholder);
    editor.appendChild(line);
    return line;
}

function handleColoredEditorKey(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        const sel = window.getSelection();
        const line = sel.anchorNode.closest ? sel.anchorNode.closest('.ce-line') : sel.anchorNode.parentElement.closest('.ce-line');
        if (!line) return;
        const newLine = document.createElement('div');
        newLine.className = 'ce-line';
        newLine.contentEditable = 'true';
        line.after(newLine);
        newLine.focus();
        recolorAllLines.call(line.parentElement);
    } else if (e.key === 'Backspace') {
        const sel = window.getSelection();
        const line = sel.anchorNode.closest ? sel.anchorNode.closest('.ce-line') : sel.anchorNode.parentElement.closest('.ce-line');
        if (!line) return;
        if (sel.anchorOffset === 0 && line.previousElementSibling) {
            e.preventDefault();
            const prev = line.previousElementSibling;
            const prevLen = prev.textContent.length;
            prev.textContent += line.textContent;
            line.remove();
            // Set cursor at merge point
            const range = document.createRange();
            const textNode = prev.firstChild || prev;
            range.setStart(textNode, Math.min(prevLen, textNode.length || 0));
            range.collapse(true);
            sel.removeAllRanges();
            sel.addRange(range);
            recolorAllLines.call(prev.parentElement);
        }
    }
}

function handleColoredPaste(e) {
    e.preventDefault();
    const text = e.clipboardData.getData('text/plain');
    const lines = text.split('\\n');
    const sel = window.getSelection();
    const currentLine = sel.anchorNode.closest ? sel.anchorNode.closest('.ce-line') : sel.anchorNode.parentElement.closest('.ce-line');
    if (!currentLine) return;
    // Insert first line at cursor
    document.execCommand('insertText', false, lines[0]);
    // Insert remaining lines as new elements
    let insertAfter = currentLine;
    for (let i = 1; i < lines.length; i++) {
        const newLine = document.createElement('div');
        newLine.className = 'ce-line';
        newLine.contentEditable = 'true';
        newLine.textContent = lines[i];
        insertAfter.after(newLine);
        insertAfter = newLine;
    }
    recolorAllLines.call(this);
}

function recolorAllLines() {
    const editor = this.closest ? this.closest('.colored-editor') : this;
    if (!editor) return;
    const offset = editor._colorOffset || 0;
    const lines = editor.querySelectorAll('.ce-line');
    let colorIdx = 0;
    lines.forEach((line) => {
        const isEmpty = line.textContent.trim() === '';
        line.classList.toggle('empty', isEmpty);
        if (isEmpty) {
            line.style.borderLeftColor = 'rgba(100,120,150,.3)';
        } else {
            line.style.borderLeftColor = colors[(colorIdx + offset) % colors.length];
            colorIdx++;
        }
    });
}

function getColoredEditorValue(id) {
    const editor = document.getElementById(id);
    if (!editor) return '';
    return Array.from(editor.querySelectorAll('.ce-line'))
        .map(line => line.textContent)
        .join('\\n');
}

function getColoredEditorLines(id) {
    // Returns array of non-empty lines directly (more robust than join+split)
    const editor = document.getElementById(id);
    if (!editor) return [];
    return Array.from(editor.querySelectorAll('.ce-line'))
        .map(line => line.textContent.trim())
        .filter(text => text.length > 0);
}

// ════════════════════════════════════════════════════════════════════════════════
// SYNTAX EDITOR - Highlight {text} and {question} placeholders
// ════════════════════════════════════════════════════════════════════════════════
function initSyntaxEditor(id, initialText) {
    const editor = document.getElementById(id);
    if (!editor) return;
    editor.contentEditable = 'true';
    editor.innerHTML = highlightSyntax(initialText || '');
    editor.addEventListener('input', handleSyntaxInput);
    editor.addEventListener('paste', handleSyntaxPaste);
    editor.addEventListener('keydown', handleSyntaxKeydown);
}

function highlightSyntax(text) {
    // Escape HTML first, preserve newlines
    const escaped = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');
    // Highlight {text}, {question}, ANSWER:, and <answer> tags
    return escaped
        .replace(/\\{text\\}/g, '<span class="hl-text">{text}</span>')
        .replace(/\\{question\\}/g, '<span class="hl-question">{question}</span>')
        .replace(/ANSWER:/gi, '<span class="hl-answer">ANSWER:</span>')
        .replace(/&lt;answer&gt;/gi, '<span class="hl-answer">&lt;answer&gt;</span>')
        .replace(/&lt;\\/answer&gt;/gi, '<span class="hl-answer">&lt;/answer&gt;</span>');
}

function handleSyntaxKeydown(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        document.execCommand('insertLineBreak');
    }
}

function handleSyntaxInput(e) {
    const editor = e.target;
    // Debounce re-highlighting to avoid cursor issues
    clearTimeout(editor._highlightTimer);
    editor._highlightTimer = setTimeout(() => {
        const sel = window.getSelection();
        if (!sel.rangeCount) return;
        const range = sel.getRangeAt(0);
        // Get cursor position as text offset
        const preRange = document.createRange();
        preRange.selectNodeContents(editor);
        preRange.setEnd(range.startContainer, range.startOffset);
        const cursorPos = preRange.toString().length;
        // Get text preserving newlines
        const text = editor.innerText;
        editor.innerHTML = highlightSyntax(text);
        // Restore cursor
        restoreCursor(editor, cursorPos);
    }, 150);
}

function handleSyntaxPaste(e) {
    e.preventDefault();
    const text = e.clipboardData.getData('text/plain');
    document.execCommand('insertText', false, text);
}

function restoreCursor(editor, pos) {
    const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT, null, false);
    let charCount = 0;
    let node;
    while (node = walker.nextNode()) {
        const len = node.textContent.length;
        if (charCount + len >= pos) {
            const range = document.createRange();
            range.setStart(node, Math.min(pos - charCount, len));
            range.collapse(true);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            return;
        }
        charCount += len;
    }
    // If we couldn't find position, put cursor at end
    const range = document.createRange();
    range.selectNodeContents(editor);
    range.collapse(false);
    const sel = window.getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
}

function getSyntaxEditorValue(id) {
    const editor = document.getElementById(id);
    return editor ? editor.innerText : '';
}

function initAllSyntaxEditors() {
    document.querySelectorAll('.syntax-editor[data-init]').forEach(editor => {
        const initText = editor.getAttribute('data-init');
        initSyntaxEditor(editor.id, initText);
    });
}

// ════════════════════════════════════════════════════════════════════════════════
// DRAGGABLE PANELS
// ════════════════════════════════════════════════════════════════════════════════
function makeDraggable(el){
    let isDragging=false,startX,startY,startLeft,startTop;
    el.addEventListener('mousedown',e=>{
        if(e.target.closest('.colored-editor,button,textarea,input'))return;
        isDragging=true;
        startX=e.clientX;startY=e.clientY;
        const rect=el.getBoundingClientRect();
        startLeft=rect.left;startTop=rect.top;
        el.style.right='auto';el.style.bottom='auto';
        el.style.left=startLeft+'px';el.style.top=startTop+'px';
        e.preventDefault();
    });
    document.addEventListener('mousemove',e=>{
        if(!isDragging)return;
        const dx=e.clientX-startX,dy=e.clientY-startY;
        el.style.left=Math.max(0,Math.min(innerWidth-el.offsetWidth,startLeft+dx))+'px';
        el.style.top=Math.max(0,Math.min(innerHeight-el.offsetHeight,startTop+dy))+'px';
    });
    document.addEventListener('mouseup',()=>isDragging=false);
}

// ════════════════════════════════════════════════════════════════════════════════
// INITIALIZATION
// ════════════════════════════════════════════════════════════════════════════════
function initAllColoredEditors() {
    document.querySelectorAll('.colored-editor[data-init]').forEach(editor => {
        const initText = editor.getAttribute('data-init');
        const placeholder = editor.getAttribute('data-placeholder') || '';
        const colorOffset = parseInt(editor.getAttribute('data-color-offset')) || 0;
        initColoredEditor(editor.id, initText, placeholder, colorOffset);
    });
}

loadConfig();connect();initAllColoredEditors();initAllSyntaxEditors();
makeDraggable(document.getElementById('addTextsPanel'));
makeDraggable(document.getElementById('pieChartPanel'));
makeDraggable(document.getElementById('legend'));
makeDraggable(document.getElementById('trajectoriesPanel'));
makeDraggable(document.getElementById('zoomControls'));
"""
