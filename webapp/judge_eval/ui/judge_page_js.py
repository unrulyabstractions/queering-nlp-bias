"""Judge page JavaScript: visualization and interaction."""

from __future__ import annotations


def get_judge_page_js() -> str:
    return """
// ════════════════════════════════════════════════════════════════════════════════
// JUDGE PAGE - Start & UI
// ════════════════════════════════════════════════════════════════════════════════
let judgePromptFormat = '';
let judgeModels = [];  // Array of {provider, model} objects
let judgeTemperature = 0.0;
let judgeMaxTokens = 32;
let axesFlipped = false;
let currentModelView = 'averaged';  // 'averaged' or 'provider/model' key
let resultsByModel = {};  // Stores per-model results for switching views

// Reset judge-specific state (called from resetExperiment in shared_app_js)
function resetJudgeState(){
    axesFlipped = false;
    judgePromptFormat = '';
    judgeModels = [];
    judgeTemperature = 0.0;
    judgeMaxTokens = 32;
    currentModelView = 'averaged';
    resultsByModel = {};
}

// Reset Judge LLM settings to defaults
function resetJudgeSettings(){
    // Reset temperature
    const tempSlider = document.getElementById('judgeTemperatureConfig');
    const tempVal = document.getElementById('judgeTempValConfig');
    if(tempSlider){ tempSlider.value = 0.0; }
    if(tempVal){ tempVal.textContent = '0.0'; }

    // Reset max tokens
    const tokensSlider = document.getElementById('judgeMaxTokensConfig');
    const tokensVal = document.getElementById('judgeMaxTokensVal');
    if(tokensSlider){ tokensSlider.value = 32; }
    if(tokensVal){ tokensVal.textContent = '32'; }

    // Reset judge prompt to default
    if(typeof settings !== 'undefined' && settings.judge_prompt){
        initSyntaxEditor('judgePromptFormat', settings.judge_prompt);
    }

    // Reset questions to default
    const questionsEl = document.getElementById('judgeQuestions');
    if(questionsEl){
        const defaultQuestions = questionsEl.dataset.init || '';
        initColoredEditor('judgeQuestions', defaultQuestions.replace(/&#10;/g, '\\n'), 'Enter a question...', 6);
    }

    // Reset model selection to default (gpt-4o-mini only)
    const checkboxes = document.querySelectorAll('#judgeModelList input[type="checkbox"]');
    checkboxes.forEach(cb => {
        cb.checked = (cb.dataset.provider === 'openai' && cb.dataset.model === 'gpt-4o-mini');
    });

    console.log('Judge settings reset to defaults');
}

// Get all API keys as dict (uses the one from shared_app_js)

// Validate API keys for selected judge models
function validateJudgeApiKeys(models){
    const keys = getApiKeys();
    for(const m of models){
        if(!isLocalProvider(m.provider) && !keys[m.provider]){
            return `Configure ${m.provider} API key in Settings`;
        }
    }
    return null;
}

function startJudge(){
    // Get selected models in new format [{provider, model}]
    judgeModels=getSelectedModelsFromList('judgeModelList');
    if(!judgeModels.length){
        alert('Select at least one judge model');showSettings();return;
    }
    judgeTemperature=parseFloat(document.getElementById('judgeTemperatureConfig')?.value || 0.0);
    judgeMaxTokens=parseInt(document.getElementById('judgeMaxTokensConfig')?.value || 500);
    // Validate API keys for all selected providers
    const keyError=validateJudgeApiKeys(judgeModels);
    if(keyError){alert(keyError);showSettings();return}
    const questions = getColoredEditorLines('judgeQuestions');
    if(!questions.length){alert('Add at least one question');return}
    currentMode='judge';
    judgeState.texts=[];
    judgeState.results=[];
    judgeState.questions=questions;
    judgePromptFormat=getSyntaxEditorValue('judgePromptFormat')||settings.judge_prompt;
    document.getElementById('judgeConfig').classList.remove('visible');
    showJudgeUI();
    initJudgeViz();
}

function addMoreTexts(){
    const t=getColoredEditorLines('addTextsInput');
    if(!t.length){console.log('No texts to evaluate');return}
    console.log('Judge API call:',{models:judgeModels,texts:t.length,questions:judgeState.questions.length});
    // Validate API keys for all selected providers
    const keyError=validateJudgeApiKeys(judgeModels);
    if(keyError){alert(keyError);return}
    // Calculate offset BEFORE adding new texts
    const textOffset=judgeState.texts.length;
    judgeState.texts.push(...t);
    // Set flag to prevent results reset on 'started' event
    judgeAddingMore=true;
    // Immediately redraw to show pending state
    drawJudgeHeatmap();
    document.getElementById('liveBadge').classList.remove('paused');
    document.getElementById('liveText').textContent='Judging';
    ws.send(JSON.stringify({
        action:'start_judge',
        api_keys:getApiKeys(),
        texts:t,
        text_offset:textOffset,
        questions:judgeState.questions,
        settings:{...settings,judge_model:judgeModels,judge_temperature:judgeTemperature,judge_max_tokens:judgeMaxTokens,judge_prompt:judgePromptFormat}
    }));
    // Clear the editor
    initColoredEditor('addTextsInput','','Enter text to judge...',4);
}

function showJudgeUI(){
    // Show UI elements (settings only on landing page)
    ['controlButtons','legend','addTextsPanel','pieChartPanel'].forEach(id=>document.getElementById(id).classList.add('visible'));
    // Explicitly hide tree/dynamics-only elements
    ['zoomControls','modeToggle','viewToggle','evolutionModeToggle'].forEach(id=>document.getElementById(id).classList.remove('visible'));
    document.getElementById('flipAxesBtn').style.display='inline-block';
    // Model toggle shown when multiple models
    updateModelToggle();
    drawPieChart();
}

function flipAxes(){
    axesFlipped=!axesFlipped;
    drawJudgeHeatmap();
}

// ════════════════════════════════════════════════════════════════════════════════
// JUDGE PAGE - Visualization
// ════════════════════════════════════════════════════════════════════════════════
function initJudgeViz(){
    judgeState.results=[];
    setupJudgeLegend();
    drawJudgeHeatmap();
    document.getElementById('statValue1').textContent='0';
    document.getElementById('statValue2').textContent='0';
}

function updateJudgeResult(d){
    // Merge new results - find by text_idx to avoid duplicates
    const existingIds=new Set(judgeState.results.map(r=>r.text_idx));
    d.all_results.forEach(r=>{
        if(!existingIds.has(r.text_idx)){
            judgeState.results.push(r);
        }
    });
    // Sort by text_idx for consistent display
    judgeState.results.sort((a,b)=>a.text_idx-b.text_idx);

    // Store per-model results for view switching
    if(d.results_by_model){
        Object.keys(d.results_by_model).forEach(modelKey=>{
            resultsByModel[modelKey]=d.results_by_model[modelKey];
        });
        updateModelToggle();
    }

    document.getElementById('progressBar').style.width=(d.progress*100)+'%';
    drawJudgeHeatmap();
    drawPieChart();
}

function updateModelToggle(){
    const toggle=document.getElementById('judgeModelToggle');
    if(!toggle)return;

    // Show toggle based on selected judgeModels (not just results)
    // Only show toggle if multiple models selected
    if(!judgeModels||judgeModels.length<=1){
        toggle.classList.remove('visible');
        return;
    }

    toggle.classList.add('visible');
    toggle.innerHTML='';

    // Add "Averaged" option
    const avgBtn=document.createElement('button');
    avgBtn.className='mode-btn'+(currentModelView==='averaged'?' active':'');
    avgBtn.textContent='Averaged';
    avgBtn.onclick=()=>switchModelView('averaged');
    toggle.appendChild(avgBtn);

    // Add per-model options based on judgeModels (selected models)
    judgeModels.forEach(m=>{
        const key=`${m.provider}/${m.model}`;
        const btn=document.createElement('button');
        btn.className='mode-btn'+(currentModelView===key?' active':'');
        // Parse provider/model key - extract model name and mode
        const provider=m.provider||'';
        const modelName=m.model||'';
        const shortName=modelName.split('/').pop();  // Just the model name
        // Extract mode from provider (huggingface_base -> Base, etc.)
        let modeLabel='';
        if(provider.includes('_base'))modeLabel='(Base)';
        else if(provider.includes('_instruct'))modeLabel='(Instruct)';
        else if(provider.includes('_reasoning'))modeLabel='(Reasoning)';
        // Show model name, and mode on second line if applicable
        if(modeLabel){
            btn.innerHTML=`${shortName}<br><span style="font-size:10px;opacity:0.7">${modeLabel}</span>`;
        }else{
            btn.textContent=shortName;
        }
        btn.title=key;
        btn.onclick=()=>switchModelView(key);
        toggle.appendChild(btn);
    });
}

function switchModelView(modelKey){
    currentModelView=modelKey;
    updateModelToggle();
    drawJudgeHeatmap();
    drawPieChart();
}

function drawJudgeHeatmap(){
    const texts=judgeState.texts||[],qs=judgeState.questions||[];
    if(!texts.length||!qs.length)return;

    // Select which results to display based on currentModelView
    let res;
    if(currentModelView==='averaged'){
        res=judgeState.results||[];
    }else{
        res=resultsByModel[currentModelView]||[];
    }

    // Build a map of text_idx -> result for quick lookup
    const resultMap={};
    res.forEach(r=>{resultMap[r.text_idx]=r;});

    const w=innerWidth,h=innerHeight-120;
    d3.select('#canvas').selectAll('*').remove();
    const svg=d3.select('#canvas').attr('width',w).attr('height',h);
    // Pastel color scale
    const cs=d3.scaleLinear().domain([0,.5,1]).range(['#e8a0a8','#c490d1','#7ec8c8']);

    // Processing indicator for pending cells
    const pendingText='...';

    if(axesFlipped){
        // Flipped: Questions as rows, Texts as columns
        const maxTextLen=Math.max(...texts.map(t=>t.length));
        const m={top:80,right:80,bottom:80,left:280};
        const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
        const cw=w-m.left-m.right,ch=h-m.top-m.bottom;
        const cellW=Math.min(120,cw/Math.max(texts.length,2)),cellH=Math.min(50,ch/qs.length);
        const totalW=cellW*texts.length,totalH=cellH*qs.length;
        const offsetX=(cw-totalW)/2,offsetY=Math.max(0,(ch-totalH)/4);

        qs.forEach((q,qi)=>{
            texts.forEach((txt,ti)=>{
                const r=resultMap[ti];
                const x=offsetX+ti*cellW,y=offsetY+qi*cellH;
                if(r&&r.scores){
                    const s=(r.scores||[])[qi];
                    const isError = s === null || s === undefined;
                    const scoreVal = isError ? 0 : s;
                    const fillColor = isError ? '#9a8baa' : cs(scoreVal);
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill',fillColor).attr('rx',8).attr('opacity',0).on('mouseover',e=>showJudgeTooltip(e,r,qi,s)).on('mouseout',hideTooltip).transition().duration(300).delay(qi*30+ti*15).attr('opacity',.85);
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill',isError?'#fff':(scoreVal>.6?'#4a3f5c':'#fff')).attr('font-size','13px').attr('font-weight','600').attr('pointer-events','none').text(isError?'ERROR':scoreVal.toFixed(2));
                }else{
                    // Pending state - gray with single word
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill','#e8e4ec').attr('rx',8).attr('opacity',.6);
                    const word = pendingText;
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill','#9a8baa').attr('font-size','12px').attr('font-style','italic').attr('opacity',.8).text(word);
                }
            });
            // Question labels on left (with tooltip for full text)
            const qLabel=g.append('text').attr('x',offsetX-12).attr('y',offsetY+qi*cellH+cellH/2+4).attr('text-anchor','end').attr('fill',colors[qi%colors.length]).attr('font-size','12px').attr('font-weight','600').attr('cursor','default');
            qLabel.append('title').text(q);
            qLabel.append('tspan').text(q.slice(0,35)+(q.length>35?'...':''));
        });
        // Text labels on top (with tooltip for full text)
        texts.forEach((txt,i)=>{
            const maxLen=Math.floor(cellW/6);
            const tLabel=svg.append('text').attr('x',m.left+offsetX+i*cellW+cellW/2).attr('y',m.top+offsetY-12).attr('text-anchor','middle').attr('fill','#7a6b8a').attr('font-size','10px').attr('cursor','default');
            tLabel.append('title').text(txt);
            tLabel.append('tspan').text(txt.length>maxLen?txt.slice(0,maxLen)+'...':txt);
        });
    }else{
        // Normal: Texts as rows, Questions as columns
        const maxTextLen=Math.max(...texts.map(t=>t.length));
        const textWidth=Math.min(500,Math.max(200,maxTextLen*7));
        const m={top:100,right:80,bottom:80,left:textWidth+40};
        const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
        const cw=w-m.left-m.right,ch=h-m.top-m.bottom;
        const cellW=Math.min(180,cw/qs.length),cellH=Math.min(50,ch/Math.max(texts.length,2));
        const totalW=cellW*qs.length,totalH=cellH*texts.length;
        const offsetX=(cw-totalW)/2,offsetY=Math.max(0,(ch-totalH)/4);

        texts.forEach((txt,ti)=>{
            const r=resultMap[ti];
            qs.forEach((q,ci)=>{
                const x=offsetX+ci*cellW,y=offsetY+ti*cellH;
                if(r&&r.scores){
                    const s=(r.scores||[])[ci];
                    const isError = s === null || s === undefined;
                    const scoreVal = isError ? 0 : s;
                    const fillColor = isError ? '#9a8baa' : cs(scoreVal);
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill',fillColor).attr('rx',8).attr('opacity',0).on('mouseover',e=>showJudgeTooltip(e,r,ci,s)).on('mouseout',hideTooltip).transition().duration(300).delay(ti*30+ci*15).attr('opacity',.85);
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill',isError?'#fff':(scoreVal>.6?'#4a3f5c':'#fff')).attr('font-size','13px').attr('font-weight','600').attr('pointer-events','none').text(isError?'ERROR':scoreVal.toFixed(2));
                }else{
                    // Pending state - gray with processing indicator
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill','#e8e4ec').attr('rx',8).attr('opacity',.6);
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill','#9a8baa').attr('font-size','12px').attr('font-style','italic').attr('opacity',.8).text(pendingText);
                }
            });
            const rowLabel=g.append('text').attr('x',offsetX-12).attr('y',offsetY+ti*cellH+cellH/2+4).attr('text-anchor','end').attr('fill','#7a6b8a').attr('font-size','12px').attr('cursor','default');
            rowLabel.append('title').text(txt);
            rowLabel.append('tspan').text(txt);
        });

        qs.forEach((q,i)=>{
            const maxLen=Math.floor(cellW/7);
            const colLabel=svg.append('text').attr('x',m.left+offsetX+i*cellW+cellW/2).attr('y',m.top+offsetY-12).attr('text-anchor','middle').attr('fill',colors[i%colors.length]).attr('font-size','11px').attr('font-weight','600').attr('cursor','default');
            colLabel.append('title').text(q);
            colLabel.append('tspan').text(q.length>maxLen?q.slice(0,maxLen)+'...':q);
        });
    }
}


function showJudgeTooltip(e,r,qi,s){
    const q=judgeState.questions[qi]||'Q'+(qi+1);
    const raw=r.raw_responses?r.raw_responses[qi]:'N/A';
    // Convert logprob to probability percentage
    const prob=r.logprobs&&r.logprobs[qi]!==null?(Math.exp(r.logprobs[qi])*100).toFixed(1)+'%':'N/A';
    const isError = s === null || s === undefined;
    const scoreDisplay = isError ? '<span style="color:#e8a0a8;font-weight:700">ERROR</span>' : s.toFixed(3);
    const tt=document.getElementById('tooltip');
    tt.innerHTML=`<div class="tooltip-title">Text ${r.text_idx+1}</div><div class="tooltip-row" style="max-width:350px;white-space:normal;margin-bottom:10px"><span>${r.full_text||r.text}</span></div><div class="tooltip-row"><span>${q}</span></div><div class="tooltip-row"><span>Score</span><span class="tooltip-value">${scoreDisplay}</span></div><div class="tooltip-row"><span>Confidence</span><span class="tooltip-value">${prob}</span></div><div class="tooltip-raw">${raw}</div>`;
    tt.style.left=Math.min(e.pageX+20,innerWidth-420)+'px';
    tt.style.top=Math.max(10,e.pageY-100)+'px';
    tt.classList.add('visible');
}

function setupJudgeLegend(){
    const leg=document.getElementById('legend');
    leg.innerHTML='';
    judgeState.questions.forEach((q,i)=>{
        const it=document.createElement('div');
        it.className='legend-item';
        it.innerHTML=`<div class="legend-swatch" style="background:${colors[i%colors.length]}"></div><span>${q}</span>`;
        leg.appendChild(it);
    });
}

function drawPieChart(){
    const svg=d3.select('#pieChart');
    svg.selectAll('*').remove();
    const w=140,h=140,r=Math.min(w,h)/2-10;
    const g=svg.append('g').attr('transform',`translate(${w/2},${h/2})`);

    // Select results based on current model view
    let res;
    if(currentModelView==='averaged'){
        res=judgeState.results||[];
    }else{
        res=resultsByModel[currentModelView]||[];
    }
    const qs=judgeState.questions||[];
    if(!qs.length)return;

    // Calculate average score per question (skip errors)
    const avgs=qs.map((_,qi)=>{
        if(!res.length)return 0;
        const validScores=res.map(r=>(r.scores||[])[qi]).filter(s=>s!==null&&s!==undefined);
        if(!validScores.length)return 0;
        return validScores.reduce((a,b)=>a+b,0)/validScores.length;
    });

    // Use actual values - only questions with non-zero avg get slices
    const total=avgs.reduce((a,b)=>a+b,0);
    let data;
    if(!res.length||total===0){
        // No results or all zeros: show equal faded slices as placeholder
        data=qs.map((_,i)=>({value:1,idx:i,avg:0,placeholder:true}));
    }else{
        // Use actual averages - 0 values get no slice
        data=avgs.map((v,i)=>({value:v,idx:i,avg:v,placeholder:false})).filter(d=>d.value>0);
    }

    const pie=d3.pie().value(d=>d.value).sort(null);
    const arc=d3.arc().innerRadius(r*0.5).outerRadius(r);

    g.selectAll('path')
        .data(pie(data))
        .enter()
        .append('path')
        .attr('d',arc)
        .attr('fill',d=>colors[d.data.idx%colors.length])
        .attr('stroke','rgba(255,255,255,.5)')
        .attr('stroke-width',2)
        .style('opacity',d=>d.data.placeholder?0.3:0.85)
        .transition().duration(500)
        .attrTween('d',function(d){
            const i=d3.interpolate({startAngle:0,endAngle:0},d);
            return t=>arc(i(t));
        });

    // Center text showing total texts
    g.append('text')
        .attr('text-anchor','middle')
        .attr('dy','0.35em')
        .attr('fill','#4a3f5c')
        .attr('font-size','18px')
        .attr('font-weight','700')
        .text(res.length||'0');
}
"""
