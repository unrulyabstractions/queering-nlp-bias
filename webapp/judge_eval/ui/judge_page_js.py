"""Judge page JavaScript: visualization and interaction."""

from __future__ import annotations


def get_judge_page_js() -> str:
    return """
// ════════════════════════════════════════════════════════════════════════════════
// JUDGE PAGE - Start & UI
// ════════════════════════════════════════════════════════════════════════════════
let judgePromptFormat = '';
let judgeProvider = 'openai';
let judgeModel = 'gpt-4o-mini';
let axesFlipped = false;

function getJudgeApiKey(){
    return judgeProvider==='openai'?appConfig.openai_key:appConfig.anthropic_key;
}

function startJudge(){
    judgeProvider=document.getElementById('judgeProvider').value;
    judgeModel=document.getElementById('judgeModel').value;
    const k=getJudgeApiKey();
    if(!k){alert('Configure '+judgeProvider+' API key in Settings');showSettings();return}
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
    const apiKey=getJudgeApiKey();
    console.log('Judge API call:',{provider:judgeProvider,model:judgeModel,apiKey:apiKey?'(set)':'(MISSING)',texts:t.length,questions:judgeState.questions.length});
    if(!apiKey){alert('Configure '+judgeProvider+' API key in Settings');return}
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
        api_key:apiKey,
        texts:t,
        text_offset:textOffset,
        questions:judgeState.questions,
        settings:{...settings,judge_provider:judgeProvider,judge_model:judgeModel,judge_prompt:judgePromptFormat}
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
    document.getElementById('progressBar').style.width=(d.progress*100)+'%';
    drawJudgeHeatmap();
    drawPieChart();
}

function drawJudgeHeatmap(){
    const texts=judgeState.texts||[],res=judgeState.results||[],qs=judgeState.questions||[];
    if(!texts.length||!qs.length)return;

    // Build a map of text_idx -> result for quick lookup
    const resultMap={};
    res.forEach(r=>{resultMap[r.text_idx]=r;});

    const w=innerWidth,h=innerHeight-120;
    d3.select('#canvas').selectAll('*').remove();
    const svg=d3.select('#canvas').attr('width',w).attr('height',h);
    // Pastel color scale
    const cs=d3.scaleLinear().domain([0,.5,1]).range(['#e8a0a8','#c490d1','#7ec8c8']);

    // Rotating words to show in pending cells - one word per cell
    const pendingWords=['Robot','is','judging','you','honey','wait','thinking','hmm','soon','patience'];

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
                    const s=(r.scores||[])[qi]||0;
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill',cs(s)).attr('rx',8).attr('opacity',0).on('mouseover',e=>showJudgeTooltip(e,r,qi,s)).on('mouseout',hideTooltip).transition().duration(300).delay(qi*30+ti*15).attr('opacity',.85);
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill',s>.6?'#4a3f5c':'#fff').attr('font-size','13px').attr('font-weight','600').attr('pointer-events','none').text(s.toFixed(2));
                }else{
                    // Pending state - gray with single word
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill','#e8e4ec').attr('rx',8).attr('opacity',.6);
                    const word = pendingWords[(qi+ti) % pendingWords.length];
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill','#9a8baa').attr('font-size','12px').attr('font-style','italic').attr('opacity',.8).text(word);
                }
            });
            // Question labels on left
            g.append('text').attr('x',offsetX-12).attr('y',offsetY+qi*cellH+cellH/2+4).attr('text-anchor','end').attr('fill',colors[qi%colors.length]).attr('font-size','12px').attr('font-weight','600').text(q.slice(0,35)+(q.length>35?'...':''));
        });
        // Text labels on top
        texts.forEach((txt,i)=>{
            const maxLen=Math.floor(cellW/6);
            svg.append('text').attr('x',m.left+offsetX+i*cellW+cellW/2).attr('y',m.top+offsetY-12).attr('text-anchor','middle').attr('fill','#7a6b8a').attr('font-size','10px').text(txt.length>maxLen?txt.slice(0,maxLen)+'...':txt);
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
                    const s=(r.scores||[])[ci]||0;
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill',cs(s)).attr('rx',8).attr('opacity',0).on('mouseover',e=>showJudgeTooltip(e,r,ci,s)).on('mouseout',hideTooltip).transition().duration(300).delay(ti*30+ci*15).attr('opacity',.85);
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill',s>.6?'#4a3f5c':'#fff').attr('font-size','13px').attr('font-weight','600').attr('pointer-events','none').text(s.toFixed(2));
                }else{
                    // Pending state - gray with single word
                    g.append('rect').attr('x',x+2).attr('y',y+2).attr('width',cellW-4).attr('height',cellH-4).attr('fill','#e8e4ec').attr('rx',8).attr('opacity',.6);
                    const word = pendingWords[(ti+ci) % pendingWords.length];
                    g.append('text').attr('x',x+cellW/2).attr('y',y+cellH/2+5).attr('text-anchor','middle').attr('fill','#9a8baa').attr('font-size','12px').attr('font-style','italic').attr('opacity',.8).text(word);
                }
            });
            g.append('text').attr('x',offsetX-12).attr('y',offsetY+ti*cellH+cellH/2+4).attr('text-anchor','end').attr('fill','#7a6b8a').attr('font-size','12px').text(txt);
        });

        qs.forEach((q,i)=>{
            const maxLen=Math.floor(cellW/7);
            svg.append('text').attr('x',m.left+offsetX+i*cellW+cellW/2).attr('y',m.top+offsetY-12).attr('text-anchor','middle').attr('fill',colors[i%colors.length]).attr('font-size','11px').attr('font-weight','600').text(q.length>maxLen?q.slice(0,maxLen)+'...':q);
        });
    }
}


function showJudgeTooltip(e,r,qi,s){
    const q=judgeState.questions[qi]||'Q'+(qi+1);
    const raw=r.raw_responses?r.raw_responses[qi]:'N/A';
    // Convert logprob to probability percentage
    const prob=r.logprobs&&r.logprobs[qi]!==null?(Math.exp(r.logprobs[qi])*100).toFixed(1)+'%':'N/A';
    const tt=document.getElementById('tooltip');
    tt.innerHTML=`<div class="tooltip-title">Text ${r.text_idx+1}</div><div class="tooltip-row" style="max-width:350px;white-space:normal;margin-bottom:10px"><span>${r.full_text||r.text}</span></div><div class="tooltip-row"><span>${q}</span></div><div class="tooltip-row"><span>Score</span><span class="tooltip-value">${s.toFixed(3)}</span></div><div class="tooltip-row"><span>Confidence</span><span class="tooltip-value">${prob}</span></div><div class="tooltip-raw">${raw}</div>`;
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

    const res=judgeState.results||[],qs=judgeState.questions||[];
    if(!qs.length)return;

    // Calculate average score per question
    const avgs=qs.map((_,qi)=>{
        if(!res.length)return 0;
        const scores=res.map(r=>(r.scores||[])[qi]||0);
        return scores.reduce((a,b)=>a+b,0)/scores.length;
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
