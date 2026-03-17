"""Dynamics page JavaScript: visualization and interaction."""

from __future__ import annotations


def get_dynamics_page_js() -> str:
    return """
// ════════════════════════════════════════════════════════════════════════════════
// DYNAMICS PAGE - Start & UI
// ════════════════════════════════════════════════════════════════════════════════
function startDynamics(){
    const keys=getApiKeys();
    // Check if at least one judge model is selected
    if(!settings.judge_model || settings.judge_model.length===0){
        alert('Select at least one judge model in Settings');showSettings();return;
    }
    // Check if API keys are needed
    const genNeedsKey = !isLocalProvider(settings.gen_provider);
    const genKeyOk = !genNeedsKey || keys[settings.gen_provider];
    if(!genKeyOk){alert('Configure generation API key in Settings');showSettings();return}
    // Check all judge model providers have keys
    for(const m of settings.judge_model){
        if(!isLocalProvider(m.provider) && !keys[m.provider]){
            alert(`Configure ${m.provider} API key in Settings`);showSettings();return;
        }
    }
    currentMode='dynamics';
    const prompt = document.getElementById('dynPrompt').value;
    dynamicsState.prompt = prompt;  // Store prompt for trajectory explorer
    ws.send(JSON.stringify({
        action:'start_dynamics',
        api_keys:keys,
        prompt:prompt,
        prefill:document.getElementById('dynPrefill').value,
        continuation:document.getElementById('dynContinuation').value,
        questions:getColoredEditorLines('dynQuestions'),
        max_rounds:parseInt(document.getElementById('dynStep').value),
        settings
    }));
    document.getElementById('dynamicsConfig').classList.remove('visible');
    showDynamicsUI();
}

let showDiversity = false;

function showDynamicsUI(){
    // Show UI elements (no textDisplay - words shown on x-axis, settings only on landing page)
    ['floatingStats','viewToggle','controlButtons','legend','evolutionModeToggle','diversityToggle'].forEach(id=>document.getElementById(id).classList.add('visible'));
    // Explicitly hide tree-only elements
    ['zoomControls','modeToggle'].forEach(id=>document.getElementById(id).classList.remove('visible'));
    document.getElementById('statLabel1').textContent='Positions';
    document.getElementById('statLabel2').textContent='API Calls';
}

function toggleDiversity(){
    showDiversity = document.getElementById('diversityCheckbox').checked;
    drawDynamicsChart();
}

// ════════════════════════════════════════════════════════════════════════════════
// DYNAMICS PAGE - Visualization
// ════════════════════════════════════════════════════════════════════════════════
function initDynamicsViz(){
    console.log('📊 initDynamicsViz called');
    console.log('📊 dynamicsState:', dynamicsState);

    // Initialize from nodes - dynamicsState IS the data object (set via dynamicsState = m.data)
    const nodes = dynamicsState.nodes || [];
    console.log('📊 Nodes count:', nodes.length);

    dynamicsState.positions = nodes.map(n => ({
        position: n.depth,
        label: n.label,
        core: n.core || [],
        orientations: n.core || [],
        pull: 0,
        drift: 0,
        potential: 0,
        coreDiversity: 1.0,
    }));
    dynamicsState.questions = dynamicsState.questions || [];
    console.log('📊 Questions:', dynamicsState.questions);
    console.log('📊 Initial positions:', dynamicsState.positions.length);

    setupDynamicsLegend();
    drawDynamicsChart();
    console.log('📊 initDynamicsViz complete');
}

function updateDynamicsPosition(d){
    console.log('📊 updateDynamicsPosition:', d.node_id, 'pos:', d.position, 'progress:', (d.progress*100).toFixed(1)+'%');

    // Store positions with all system data
    dynamicsState.positions = (d.all_positions || []).map(p => ({
        ...p,
        // Per-dimension standard deviation
        orientStd: p.orient_std || [],
        // Compute drift and potential systems (vectors, not scalars)
        driftSystem: (p.prefix_system || []).map((v, i) => v - (p.initial_prefix?.[i] || 0)),
        potentialSystem: (p.final_prefix || []).map((v, i) => v - (p.prefix_system?.[i] || 0)),
        // Core diversity (effective number of structures)
        coreDiversity: p.core_diversity || 1.0,
    }));

    console.log('📊 Positions updated:', dynamicsState.positions.length,
        'with core:', dynamicsState.positions.filter(p => p.core?.length).length);

    dynamicsState.total_api_calls = d.total_api_calls || 0;
    if (d.total_errors !== undefined) dynamicsState.total_errors = d.total_errors;
    document.getElementById('statValue1').textContent = dynamicsState.positions.length;
    document.getElementById('statValue2').textContent = dynamicsState.total_api_calls;
    document.getElementById('progressBar').style.width = ((d.progress || 0) * 100) + '%';
    // Show error count if there are errors
    const errorCard = document.getElementById('errorStatCard');
    const errorVal = document.getElementById('statErrors');
    const totalErrors = dynamicsState.total_errors || 0;
    if (totalErrors > 0) {
        errorCard.style.display = 'block';
        errorVal.textContent = totalErrors;
    } else {
        errorCard.style.display = 'none';
    }

    const t = dynamicsState.continuation, p = d.position;
    if (t && p !== undefined) {
        document.getElementById('textContent').innerHTML = `<span class="text-highlight">${t.slice(0,p)}</span>${t.slice(p)}`;
    }

    drawDynamicsChart();
}

function drawDynamicsChart(){
    console.log('📊 drawDynamicsChart called, showMagnitudes:', showMagnitudes);

    const w=innerWidth,h=innerHeight-120,m={top:100,right:120,bottom:120,left:100};
    d3.select('#canvas').selectAll('*').remove();
    const svg=d3.select('#canvas').attr('width',w).attr('height',h);
    const g=svg.append('g').attr('transform',`translate(${m.left},${m.top})`);
    const pos=dynamicsState.positions||[];

    console.log('📊 Positions to plot:', pos.length);
    if(!pos.length){
        console.log('📊 No positions to plot, returning');
        return;
    }

    // Log sample data for debugging
    if(pos.length > 0) {
        const sample = pos[0];
        console.log('📊 Sample position data:', {
            position: sample.position,
            label: sample.label,
            orientations: sample.orientations,
            pull: sample.pull,
            drift: sample.drift,
            potential: sample.potential
        });
    }
    const qs=dynamicsState.questions||[];
    const cw=w-m.left-m.right,ch=h-m.top-m.bottom;

    // Use point scale for word labels on x-axis
    const labels = pos.map(d => d.label || '');
    const x = d3.scalePoint().domain(labels).range([0, cw]).padding(0.1);
    const xByPos = d => x(d.label || '');

    // Dynamic font size based on number of words and chart width
    const wordCount = pos.length;
    const spacePerWord = cw / Math.max(wordCount, 1);
    // Scale font: 12px for few words, down to 6px for many words
    const baseFontSize = Math.max(6, Math.min(12, spacePerWord * 0.8));
    const labelFontSize = `${baseFontSize}px`;

    // Calculate prefill word count for dimming x-axis labels
    const prefillText = dynamicsState.prefill || '';
    const prefillWordCount = prefillText ? prefillText.split(' ').filter(w => w).length : 0;

    if(showMagnitudes){
        const y=d3.scaleLinear().domain([0,d3.max(pos,d=>Math.max(d.pull,d.drift,d.potential))*1.1]).range([ch,0]);
        g.append('g').attr('class','grid').selectAll('line').data(y.ticks(5)).enter().append('line').attr('x1',0).attr('x2',cw).attr('y1',d=>y(d)).attr('y2',d=>y(d));
        [{k:'pull',c:'#E67E22'},{k:'drift',c:'#9B59B6'},{k:'potential',c:'#3498DB'}].forEach((mt,i)=>{
            if(highlightedMagnitude!==null&&highlightedMagnitude!==i)return;
            const ln=d3.line().x(xByPos).y(d=>y(d[mt.k])).curve(d3.curveMonotoneX);
            const ar=d3.area().x(xByPos).y0(ch).y1(d=>y(d[mt.k])).curve(d3.curveMonotoneX);
            g.append('path').datum(pos).attr('class','chart-area').attr('d',ar).attr('fill',mt.c);
            g.append('path').datum(pos).attr('class','chart-line').attr('d',ln).attr('stroke',mt.c);
            g.selectAll(`.dot-${mt.k}`).data(pos).enter().append('circle').attr('class','chart-dot').attr('cx',xByPos).attr('cy',d=>y(d[mt.k])).attr('r',5).attr('fill',mt.c);
        });
        // X-axis with word labels (dynamic font size, prefill dimmed)
        const xAxis = g.append('g').attr('class','axis').attr('transform',`translate(0,${ch})`).call(d3.axisBottom(x));
        xAxis.selectAll('text').attr('transform','rotate(-45)').attr('text-anchor','end').attr('dx','-0.5em').attr('dy','0.5em').style('font-size', labelFontSize)
            .style('fill', (d, i) => i < prefillWordCount ? '#9a8baa' : '#4a3f5c')
            .style('font-style', (d, i) => i < prefillWordCount ? 'italic' : 'normal');
        g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5));

        // Draw diversity on second y-axis if enabled - RAINBOW TREATMENT (also in magnitudes)
        if(showDiversity && pos.some(d => d.coreDiversity > 0)) {
            const maxDiversity = Math.max(...pos.map(d => d.coreDiversity || 1), 3);
            const y2 = d3.scaleLinear().domain([1, maxDiversity * 1.1]).range([ch, 0]);

            // Rainbow gradient for line
            const rainbowGrad = svg.append('defs').append('linearGradient').attr('id','rainbowLineMag').attr('x1','0%').attr('y1','0%').attr('x2','100%').attr('y2','0%');
            rainbowGrad.append('stop').attr('offset','0%').attr('stop-color','#FF6B9D');
            rainbowGrad.append('stop').attr('offset','20%').attr('stop-color','#E06C75');
            rainbowGrad.append('stop').attr('offset','40%').attr('stop-color','#E5C07B');
            rainbowGrad.append('stop').attr('offset','60%').attr('stop-color','#98C379');
            rainbowGrad.append('stop').attr('offset','80%').attr('stop-color','#56B6C2');
            rainbowGrad.append('stop').attr('offset','100%').attr('stop-color','#C678DD');

            // Glow filter
            const glowFilter = svg.append('defs').append('filter').attr('id','diversityGlowMag').attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
            glowFilter.append('feGaussianBlur').attr('stdDeviation','5').attr('result','blur');
            glowFilter.append('feComposite').attr('in','SourceGraphic').attr('in2','blur').attr('operator','over');

            // Diversity line
            const divLine = d3.line().x(xByPos).y(d => y2(d.coreDiversity || 1)).curve(d3.curveMonotoneX);
            g.append('path').datum(pos).attr('d',divLine).attr('stroke','url(#rainbowLineMag)').attr('stroke-width',12).attr('opacity',0.35).attr('fill','none').attr('filter','url(#diversityGlowMag)');
            g.append('path').datum(pos).attr('d',divLine).attr('stroke','url(#rainbowLineMag)').attr('stroke-width',4).attr('fill','none');

            // Diversity dots
            const dotColors = ['#FF6B9D','#E06C75','#E5C07B','#98C379','#56B6C2','#C678DD'];
            g.selectAll('.diversity-dot').data(pos).enter().append('circle').attr('cx',xByPos).attr('cy',d=>y2(d.coreDiversity||1)).attr('r',7).attr('fill',(d,i)=>dotColors[i%dotColors.length]).attr('stroke','white').attr('stroke-width',2).attr('filter','url(#diversityGlowMag)');

            // Second y-axis
            const y2Axis = d3.axisRight(y2).ticks(5);
            const axisG = g.append('g').attr('class','axis diversity-axis').attr('transform',`translate(${cw},0)`).call(y2Axis);
            axisG.selectAll('text').attr('fill','#C678DD').attr('font-weight','700').attr('font-size','12px');
            axisG.selectAll('line').attr('stroke','#C678DD').attr('stroke-width',2);
            axisG.select('path').attr('stroke','#C678DD').attr('stroke-width',2);
            svg.append('text').attr('transform','rotate(90)').attr('x',h/2).attr('y',-w+22).attr('text-anchor','middle').attr('fill','#C678DD').attr('font-size','13px').attr('font-weight','700').text('🌈 DIVERSITY 🌈');
        }
    }else{
        // Series mode: plot Core, Drift, or Potential system per question
        const getSystem = (d) => {
            if(evolutionMode === 'core') return d.core || [];
            if(evolutionMode === 'drift') return d.driftSystem || [];
            if(evolutionMode === 'potential') return d.potentialSystem || [];
            return [];
        };

        // Compute mean and std for error bars at each position
        const computeStats = (d) => {
            const sys = getSystem(d);
            if(!sys.length) return {mean: 0, std: 0};
            const mean = sys.reduce((a,b) => a+b, 0) / sys.length;
            const variance = sys.reduce((a,v) => a + (v-mean)**2, 0) / sys.length;
            return {mean, std: Math.sqrt(variance)};
        };

        // For drift/potential, values can be negative
        const isDelta = evolutionMode !== 'core';
        let yMin = 0, yMax = 1;
        if(isDelta) {
            pos.forEach(d => {
                const sys = getSystem(d);
                sys.forEach(v => {
                    if(v < yMin) yMin = v;
                    if(v > yMax) yMax = v;
                });
            });
            // Symmetric around 0 for delta modes
            const absMax = Math.max(Math.abs(yMin), Math.abs(yMax), 0.1);
            yMin = -absMax;
            yMax = absMax;
        }

        const y=d3.scaleLinear().domain([yMin, yMax]).range([ch,0]);

        if(isDelta) {
            // Grid lines including zero
            g.append('g').attr('class','grid').selectAll('line').data(y.ticks(5)).enter().append('line').attr('x1',0).attr('x2',cw).attr('y1',d=>y(d)).attr('y2',d=>y(d));
            // Zero line
            g.append('line').attr('x1',0).attr('x2',cw).attr('y1',y(0)).attr('y2',y(0)).attr('stroke','rgba(180,160,200,.4)').attr('stroke-width',1);
        } else {
            g.append('g').attr('class','grid').selectAll('line').data([0,.25,.5,.75,1]).enter().append('line').attr('x1',0).attr('x2',cw).attr('y1',d=>y(d)).attr('y2',d=>y(d));
        }

        // Draw per-dimension error bars (std per question) - thin, transparent
        if(evolutionMode === 'core') {
            qs.forEach((q, i) => {
                if(highlightedQuestion !== null && highlightedQuestion !== i) return;
                const color = colors[i % colors.length];
                pos.forEach(d => {
                    const std = (d.orientStd || [])[i] || 0;
                    if(std < 0.005) return;  // Skip tiny error bars
                    const val = (getSystem(d)[i] || 0);
                    const xPos = xByPos(d);
                    const yTop = y(Math.min(yMax, val + std));
                    const yBottom = y(Math.max(yMin, val - std));
                    // Thin error bar line
                    g.append('line')
                        .attr('x1', xPos).attr('x2', xPos)
                        .attr('y1', yTop).attr('y2', yBottom)
                        .attr('stroke', color).attr('stroke-width', 1.5)
                        .attr('opacity', 0.25);
                    // Small caps at top and bottom
                    g.append('line')
                        .attr('x1', xPos - 3).attr('x2', xPos + 3)
                        .attr('y1', yTop).attr('y2', yTop)
                        .attr('stroke', color).attr('stroke-width', 1)
                        .attr('opacity', 0.2);
                    g.append('line')
                        .attr('x1', xPos - 3).attr('x2', xPos + 3)
                        .attr('y1', yBottom).attr('y2', yBottom)
                        .attr('stroke', color).attr('stroke-width', 1)
                        .attr('opacity', 0.2);
                });
            });
        }

        qs.forEach((q,i)=>{
            if(highlightedQuestion!==null&&highlightedQuestion!==i)return;
            const ln=d3.line().x(xByPos).y(d=>y(getSystem(d)[i]||0)).defined(d=>getSystem(d)[i]!==undefined).curve(d3.curveMonotoneX);
            g.append('path').datum(pos).attr('class','chart-line').attr('d',ln).attr('stroke',colors[i%colors.length]);
            g.selectAll(`.dot-${i}`).data(pos.filter(d=>getSystem(d)[i]!==undefined)).enter().append('circle').attr('class','chart-dot').attr('cx',xByPos).attr('cy',d=>y(getSystem(d)[i]||0)).attr('r',5).attr('fill',colors[i%colors.length]);
        });

        // Draw diversity on second y-axis if enabled - RAINBOW TREATMENT
        if(showDiversity && pos.some(d => d.coreDiversity > 0)) {
            const maxDiversity = Math.max(...pos.map(d => d.coreDiversity || 1), qs.length);
            const y2 = d3.scaleLinear().domain([1, maxDiversity * 1.1]).range([ch, 0]);

            // Rainbow gradient for line
            const rainbowGrad = svg.append('defs').append('linearGradient').attr('id','rainbowLine').attr('x1','0%').attr('y1','0%').attr('x2','100%').attr('y2','0%');
            rainbowGrad.append('stop').attr('offset','0%').attr('stop-color','#FF6B9D');
            rainbowGrad.append('stop').attr('offset','20%').attr('stop-color','#E06C75');
            rainbowGrad.append('stop').attr('offset','40%').attr('stop-color','#E5C07B');
            rainbowGrad.append('stop').attr('offset','60%').attr('stop-color','#98C379');
            rainbowGrad.append('stop').attr('offset','80%').attr('stop-color','#56B6C2');
            rainbowGrad.append('stop').attr('offset','100%').attr('stop-color','#C678DD');

            // Rainbow area fill
            const divArea = d3.area().x(xByPos).y0(ch).y1(d => y2(d.coreDiversity || 1)).curve(d3.curveMonotoneX);
            const areaGrad = svg.append('defs').append('linearGradient').attr('id','rainbowArea').attr('x1','0%').attr('y1','0%').attr('x2','100%').attr('y2','0%');
            areaGrad.append('stop').attr('offset','0%').attr('stop-color','#FF6B9D').attr('stop-opacity',0.15);
            areaGrad.append('stop').attr('offset','25%').attr('stop-color','#E5C07B').attr('stop-opacity',0.12);
            areaGrad.append('stop').attr('offset','50%').attr('stop-color','#98C379').attr('stop-opacity',0.12);
            areaGrad.append('stop').attr('offset','75%').attr('stop-color','#56B6C2').attr('stop-opacity',0.12);
            areaGrad.append('stop').attr('offset','100%').attr('stop-color','#C678DD').attr('stop-opacity',0.15);
            g.append('path').datum(pos).attr('d',divArea).attr('fill','url(#rainbowArea)');

            // Glow filter for line
            const glowFilter = svg.append('defs').append('filter').attr('id','diversityGlow').attr('x','-50%').attr('y','-50%').attr('width','200%').attr('height','200%');
            glowFilter.append('feGaussianBlur').attr('stdDeviation','5').attr('result','blur');
            glowFilter.append('feComposite').attr('in','SourceGraphic').attr('in2','blur').attr('operator','over');

            // Diversity line - thick, rainbow, glowing
            const divLine = d3.line().x(xByPos).y(d => y2(d.coreDiversity || 1)).curve(d3.curveMonotoneX);
            g.append('path').datum(pos).attr('class','chart-line diversity-line-glow').attr('d',divLine).attr('stroke','url(#rainbowLine)').attr('stroke-width',12).attr('opacity',0.35).attr('filter','url(#diversityGlow)');
            g.append('path').datum(pos).attr('class','chart-line diversity-line').attr('d',divLine).attr('stroke','url(#rainbowLine)').attr('stroke-width',4);

            // Diversity dots - rainbow colored based on position
            const dotColors = ['#FF6B9D','#E06C75','#E5C07B','#98C379','#56B6C2','#C678DD'];
            g.selectAll('.diversity-dot').data(pos).enter().append('circle').attr('class','chart-dot diversity-dot').attr('cx',xByPos).attr('cy',d=>y2(d.coreDiversity||1)).attr('r',7).attr('fill',(d,i)=>dotColors[i%dotColors.length]).attr('stroke','white').attr('stroke-width',2).attr('filter','url(#diversityGlow)');

            // Second y-axis on right - purple styling (end of rainbow)
            const y2Axis = d3.axisRight(y2).ticks(5);
            const axisG = g.append('g').attr('class','axis diversity-axis').attr('transform',`translate(${cw},0)`).call(y2Axis);
            axisG.selectAll('text').attr('fill','#C678DD').attr('font-weight','700').attr('font-size','12px');
            axisG.selectAll('line').attr('stroke','#C678DD').attr('stroke-width',2);
            axisG.select('path').attr('stroke','#C678DD').attr('stroke-width',2);
            svg.append('text').attr('transform','rotate(90)').attr('x',h/2).attr('y',-w+22).attr('text-anchor','middle').attr('fill','#C678DD').attr('font-size','13px').attr('font-weight','700').attr('letter-spacing','0.05em').text('🌈 DIVERSITY 🌈');
        }

        // X-axis with word labels (dynamic font size, prefill dimmed)
        const xAxis = g.append('g').attr('class','axis').attr('transform',`translate(0,${ch})`).call(d3.axisBottom(x));
        xAxis.selectAll('text').attr('transform','rotate(-45)').attr('text-anchor','end').attr('dx','-0.5em').attr('dy','0.5em').style('font-size', labelFontSize)
            .style('fill', (d, i) => i < prefillWordCount ? '#9a8baa' : '#4a3f5c')
            .style('font-style', (d, i) => i < prefillWordCount ? 'italic' : 'normal');
        g.append('g').attr('class','axis').call(d3.axisLeft(y).ticks(5));
    }
    const yLabels = {core:'Core Score/Structure', drift:'Drift (Δ from start)', potential:'Potential (Δ to end)'};
    const yLabel = showMagnitudes ? 'Magnitude (L2)' : yLabels[evolutionMode] || 'Score';
    svg.append('text').attr('transform','rotate(-90)').attr('x',-h/2).attr('y',20).attr('text-anchor','middle').attr('fill','#6b7280').text(yLabel);
}

function setupDynamicsLegend(){
    const leg=document.getElementById('legend');
    leg.innerHTML='';
    if(showMagnitudes){
        [{c:'#E67E22',l:'Pull'},{c:'#9B59B6',l:'Drift'},{c:'#3498DB',l:'Potential'}].forEach((m,i)=>{
            const it=document.createElement('div');
            it.className='legend-item';
            it.innerHTML=`<div class="legend-swatch" style="background:${m.c}"></div><span>${m.l}</span>`;
            it.onclick=()=>{
                highlightedMagnitude=highlightedMagnitude===i?null:i;
                document.querySelectorAll('.legend-item').forEach((el,j)=>el.classList.toggle('dimmed',highlightedMagnitude!==null&&j!==i));
                drawDynamicsChart();
            };
            leg.appendChild(it);
        });
    }else{
        // Series mode - show questions with current mode label
        const modeLabels = {core: 'Core', drift: 'Drift', potential: 'Potential'};
        (dynamicsState.questions||[]).forEach((q,i)=>{
            const it=document.createElement('div');
            it.className='legend-item';
            it.innerHTML=`<div class="legend-swatch" style="background:${colors[i%colors.length]}"></div><span>${q}</span>`;
            it.onclick=()=>{
                highlightedQuestion=highlightedQuestion===i?null:i;
                document.querySelectorAll('.legend-item').forEach((el,j)=>el.classList.toggle('dimmed',highlightedQuestion!==null&&j!==i));
                if (showConvergence) {
                    drawConvergenceChart();
                } else {
                    drawDynamicsChart();
                }
            };
            leg.appendChild(it);
        });
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// DYNAMICS PAGE - View toggle handlers
// ════════════════════════════════════════════════════════════════════════════════
let evolutionMode = 'core'; // 'core', 'drift', 'potential'

document.getElementById('btnEvolution').onclick=()=>{
    showMagnitudes=false;
    highlightedMagnitude=null;
    document.getElementById('btnEvolution').classList.add('active');
    document.getElementById('btnMagnitudes').classList.remove('active');
    document.getElementById('evolutionModeToggle').classList.add('visible');
    document.getElementById('diversityToggle').classList.add('visible');
    setupDynamicsLegend();
    drawDynamicsChart();
};

document.getElementById('btnMagnitudes').onclick=()=>{
    showMagnitudes=true;
    highlightedQuestion=null;
    document.getElementById('btnMagnitudes').classList.add('active');
    document.getElementById('btnEvolution').classList.remove('active');
    document.getElementById('evolutionModeToggle').classList.remove('visible');
    // Keep diversity toggle visible in magnitudes too
    document.getElementById('diversityToggle').classList.add('visible');
    setupDynamicsLegend();
    drawDynamicsChart();
};

document.getElementById('btnEvolutionCore').onclick=()=>{
    evolutionMode='core';
    document.querySelectorAll('#evolutionModeToggle .mode-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('btnEvolutionCore').classList.add('active');
    setupDynamicsLegend();
    drawDynamicsChart();
};

document.getElementById('btnEvolutionDrift').onclick=()=>{
    evolutionMode='drift';
    document.querySelectorAll('#evolutionModeToggle .mode-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('btnEvolutionDrift').classList.add('active');
    setupDynamicsLegend();
    drawDynamicsChart();
};

document.getElementById('btnEvolutionPotential').onclick=()=>{
    evolutionMode='potential';
    document.querySelectorAll('#evolutionModeToggle .mode-btn').forEach(b=>b.classList.remove('active'));
    document.getElementById('btnEvolutionPotential').classList.add('active');
    setupDynamicsLegend();
    drawDynamicsChart();
};

// ════════════════════════════════════════════════════════════════════════════════
// TRAJECTORY EXPLORER - Interactive word-by-word visualization
// ════════════════════════════════════════════════════════════════════════════════
let trajExplorerState = {
    words: [],
    positions: [],
    currentIdx: 0,      // Integer index for word highlighting
    currentT: 0,        // Float 0-1 for smooth slider position
    mode: 'core', // 'core', 'drift', 'potential'
    isDragging: false,
    // Pre-calculated data for smooth rendering
    positionData: [],  // Pre-computed per-position chart data
    ballColors: [],    // Pre-computed ball colors (as {r,g,b})
    rafId: null,       // requestAnimationFrame ID
    pendingT: null     // Pending float position update
};

function showTrajExplorer() {
    console.log('📊 showTrajExplorer called');
    console.log('📊 continuation:', dynamicsState.continuation?.length, 'chars');
    console.log('📊 positions:', dynamicsState.positions?.length);

    if (!dynamicsState.continuation || !dynamicsState.positions?.length) {
        showToast('warning', 'No Data', 'Complete dynamics analysis first');
        return;
    }

    // Parse words from continuation
    const text = dynamicsState.continuation;
    const wordRegex = /\S+/g;
    const words = [];
    let match;
    while ((match = wordRegex.exec(text)) !== null) {
        words.push({ text: match[0], start: match.index, end: match.index + match[0].length });
    }

    trajExplorerState.words = words;
    trajExplorerState.positions = dynamicsState.positions;
    trajExplorerState.currentIdx = 0;
    trajExplorerState.currentT = 0;
    trajExplorerState.mode = 'core';

    // Pre-calculate all position data and ball colors for smooth scrubbing
    precomputeTrajData();

    // Display prompt at top - MUST BE VISIBLE
    const promptEl = document.getElementById('trajExplorerPrompt');
    const promptText = dynamicsState.prompt || 'No prompt set';
    promptEl.innerHTML = '<strong style="color:#2a6868">Prompt:</strong> ' + promptText;
    promptEl.style.display = 'block';

    // Build text HTML
    const textEl = document.getElementById('trajExplorerText');
    textEl.innerHTML = words.map((w, i) =>
        `<span class="word ${i === 0 ? 'current' : 'future'}" data-idx="${i}">${w.text}</span> `
    ).join('');

    // Add click handlers to words
    textEl.querySelectorAll('.word').forEach(el => {
        el.onclick = () => setTrajPosition(parseInt(el.dataset.idx));
    });

    // Setup slider
    setupTrajSlider();

    // Setup legend
    setupTrajLegend();

    // Reset mode buttons
    document.querySelectorAll('.traj-mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('trajModeCore').classList.add('active');

    // Add keyboard handler
    document.addEventListener('keydown', handleTrajKeydown);

    // Make panel visible FIRST, then draw chart after layout
    document.getElementById('trajExplorer').classList.add('visible');

    // Wait for layout before drawing chart (getBoundingClientRect needs visible element)
    requestAnimationFrame(() => {
        updateTrajExplorer();
    });
}

function handleTrajKeydown(e) {
    if (!document.getElementById('trajExplorer').classList.contains('visible')) return;

    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        setTrajPosition(trajExplorerState.currentIdx + 1);
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        setTrajPosition(trajExplorerState.currentIdx - 1);
    } else if (e.key === 'Home') {
        e.preventDefault();
        setTrajPosition(0);
    } else if (e.key === 'End') {
        e.preventDefault();
        setTrajPosition(trajExplorerState.words.length - 1);
    } else if (e.key === 'Escape') {
        hideTrajExplorer();
    } else if (e.key === '1') {
        setTrajMode('core');
    } else if (e.key === '2') {
        setTrajMode('drift');
    } else if (e.key === '3') {
        setTrajMode('potential');
    }
}

function hideTrajExplorer() {
    document.getElementById('trajExplorer').classList.remove('visible');
    document.removeEventListener('keydown', handleTrajKeydown);
}

function setupTrajSlider() {
    const track = document.getElementById('trajSliderTrack');
    const ball = document.getElementById('trajSliderBall');
    const explorer = document.getElementById('trajExplorer');

    function updateFromMouse(clientX) {
        const rect = track.getBoundingClientRect();
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        const t = x / rect.width;  // Float 0-1
        setTrajPositionSmooth(t);
    }

    track.onmousedown = (e) => {
        trajExplorerState.isDragging = true;
        explorer.classList.add('dragging');
        globalSliderDragState = {
            update: updateFromMouse,
            end: () => {
                trajExplorerState.isDragging = false;
                explorer.classList.remove('dragging');
            }
        };
        updateFromMouse(e.clientX);
        e.preventDefault();
    };

    // Position ball initially
    ball.style.left = '4px';
}

function precomputeTrajData() {
    const { positions } = trajExplorerState;
    const numQ = dynamicsState.questions?.length || 1;
    const maxNorm = Math.sqrt(numQ);

    trajExplorerState.positionData = [];
    trajExplorerState.ballColors = [];

    for (let i = 0; i < positions.length; i++) {
        const posData = positions[i] || {};
        const core = posData.core || [];
        const prefixSystem = posData.prefix_system || [];
        const initialPrefix = posData.initial_prefix || [];
        const finalPrefix = posData.final_prefix || [];

        // Pre-compute values for each mode
        const coreVals = core.length ? core : prefixSystem;
        const driftVals = prefixSystem.map((v, j) => v - (initialPrefix[j] || 0));
        const potentialVals = finalPrefix.map((v, j) => v - (prefixSystem[j] || 0));

        trajExplorerState.positionData.push({
            core: coreVals,
            drift: driftVals,
            potential: potentialVals,
            label: posData.label || ''
        });

        // Pre-compute ball color
        const l2Norm = Math.sqrt(coreVals.reduce((sum, v) => sum + v * v, 0));
        const normRatio = Math.min(l2Norm / maxNorm, 1);

        let ballColor;
        if (normRatio < 0.5) {
            const t = normRatio * 2;
            ballColor = lerpColor({r:86,g:182,b:194}, {r:255,g:107,b:157}, t);
        } else {
            const t = (normRatio - 0.5) * 2;
            ballColor = lerpColor({r:255,g:107,b:157}, {r:229,g:192,b:123}, t);
        }
        trajExplorerState.ballColors.push(ballColor);
    }

    console.log('📊 Pre-computed', trajExplorerState.positionData.length, 'positions');
}

function setTrajPosition(idx) {
    // Discrete position (from keyboard/click on word)
    const numWords = trajExplorerState.words.length;
    idx = Math.max(0, Math.min(idx, numWords - 1));
    const t = numWords > 1 ? idx / (numWords - 1) : 0;
    setTrajPositionSmooth(t);
}

function setTrajPositionSmooth(t) {
    // Smooth float position 0-1 (from slider drag)
    t = Math.max(0, Math.min(1, t));
    trajExplorerState.pendingT = t;
    if (!trajExplorerState.rafId) {
        trajExplorerState.rafId = requestAnimationFrame(applyTrajPosition);
    }
}

function applyTrajPosition() {
    trajExplorerState.rafId = null;
    const t = trajExplorerState.pendingT;
    if (t === null) return;

    const numWords = trajExplorerState.words.length;
    const numPositions = trajExplorerState.positionData.length;

    // Update float position
    trajExplorerState.currentT = t;

    // Calculate discrete word index for highlighting
    const newIdx = Math.round(t * (numWords - 1));
    const idxChanged = newIdx !== trajExplorerState.currentIdx;
    trajExplorerState.currentIdx = newIdx;

    updateTrajExplorerSmooth(t, idxChanged);
}

function setTrajMode(mode) {
    trajExplorerState.mode = mode;
    document.querySelectorAll('.traj-mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('trajMode' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
    // Force chart rebuild on mode change (different data structure)
    d3.select('#trajExplorerChart').selectAll('*').remove();
    updateTrajExplorerSmooth(trajExplorerState.currentT, false);
}

function updateTrajExplorer() {
    // Full update - used on mode change or initial load
    precomputeTrajData();
    updateTrajExplorerSmooth(trajExplorerState.currentT, true);
}

function updateTrajExplorerSmooth(t, updateWords) {
    // Smooth update with interpolation - t is float 0-1
    const { words, currentIdx, ballColors, positionData, mode } = trajExplorerState;
    const numWords = words.length;
    const numPositions = positionData.length;

    // Update word highlighting only when discrete index changes
    if (updateWords) {
        document.querySelectorAll('#trajExplorerText .word').forEach((el, i) => {
            const wasCurrent = el.classList.contains('current');
            const isCurrent = i === currentIdx;

            if (i < currentIdx) {
                if (!el.classList.contains('past')) {
                    el.classList.remove('current', 'future');
                    el.classList.add('past');
                }
            } else if (isCurrent) {
                if (!wasCurrent) {
                    el.classList.remove('past', 'future');
                    el.classList.add('current');
                }
            } else {
                if (!el.classList.contains('future')) {
                    el.classList.remove('past', 'current');
                    el.classList.add('future');
                }
            }
        });
    }

    // Update slider ball position smoothly
    const track = document.getElementById('trajSliderTrack');
    const ball = document.getElementById('trajSliderBall');
    const trackWidth = track.offsetWidth - ball.offsetWidth;
    ball.style.left = (4 + t * trackWidth) + 'px';

    // Update diversity display
    const posIdx = Math.round(t * (numPositions - 1));
    const diversity = trajExplorerState.positions[posIdx]?.coreDiversity || 1;
    document.getElementById('trajDiversityValue').textContent = diversity.toFixed(2);

    // Interpolate ball color between positions
    const floatIdx = t * (numPositions - 1);
    const idx0 = Math.floor(floatIdx);
    const idx1 = Math.min(idx0 + 1, numPositions - 1);
    const frac = floatIdx - idx0;

    const c0 = ballColors[idx0] || {r:100,g:120,b:150};
    const c1 = ballColors[idx1] || c0;
    const ballColor = lerpColor(c0, c1, frac);
    const colorStr = `rgb(${ballColor.r},${ballColor.g},${ballColor.b})`;
    ball.style.background = `radial-gradient(circle at 30% 30%, rgba(255,255,255,.9), ${colorStr})`;
    ball.style.boxShadow = `0 0 25px ${colorStr}, 0 0 50px ${colorStr}, inset 0 0 15px rgba(255,255,255,.3)`;

    // Draw bar chart with interpolated values
    drawTrajChartInterpolated(floatIdx, mode);
}

function lerpColor(c1, c2, t) {
    return {
        r: Math.round(c1.r + (c2.r - c1.r) * t),
        g: Math.round(c1.g + (c2.g - c1.g) * t),
        b: Math.round(c1.b + (c2.b - c1.b) * t)
    };
}

function drawTrajChartInterpolated(floatIdx, mode) {
    // Chart drawing with interpolation between positions
    const { positionData } = trajExplorerState;
    const numPositions = positionData.length;
    if (!numPositions) return;

    const questions = dynamicsState.questions || [];
    if (!questions.length) return;

    // Interpolate values between two adjacent positions
    const idx0 = Math.floor(floatIdx);
    const idx1 = Math.min(idx0 + 1, numPositions - 1);
    const frac = floatIdx - idx0;

    const data0 = positionData[idx0] || {};
    const data1 = positionData[idx1] || data0;
    const vals0 = data0[mode] || [];
    const vals1 = data1[mode] || vals0;

    // Interpolated values
    const values = vals0.map((v, i) => v + (vals1[i] - v) * frac);

    const titles = {
        core: 'Core (Mean Trajectory Score)',
        drift: 'Drift (Change from Initial)',
        potential: 'Potential (Distance to Final)'
    };

    const svg = d3.select('#trajExplorerChart');
    const rect = document.getElementById('trajExplorerChart').getBoundingClientRect();
    const w = rect.width || 350;
    const h = rect.height || 300;
    const m = { top: 35, right: 20, bottom: 15, left: 15 };
    const cw = w - m.left - m.right;
    const ch = h - m.top - m.bottom;

    const isDelta = mode === 'drift' || mode === 'potential';
    const maxAbs = isDelta ? Math.max(1, d3.max(values.map(Math.abs)) || 1) : 1;
    const x = d3.scaleLinear()
        .domain(isDelta ? [-maxAbs, maxAbs] : [0, 1])
        .range(isDelta ? [0, cw] : [0, cw]);
    const centerX = isDelta ? x(0) : 0;

    const y = d3.scaleBand()
        .domain(questions.map((_, i) => i))
        .range([0, ch])
        .padding(0.15);

    // Check if we can update existing bars or need to rebuild
    const existingBars = svg.selectAll('.bar-fill');
    if (existingBars.size() === questions.length) {
        // Fast path: just update bar widths and value labels
        existingBars.each(function(_, i) {
            const val = values[i] || 0;
            const bar = d3.select(this);
            if (isDelta) {
                const barX = val >= 0 ? centerX : x(val);
                const barW = Math.abs(x(val) - centerX);
                bar.attr('x', barX).attr('width', barW);
            } else {
                bar.attr('width', x(val));
            }
        });
        svg.selectAll('.bar-value').each(function(_, i) {
            const val = values[i] || 0;
            d3.select(this).text((val >= 0 ? '+' : '') + val.toFixed(2));
        });
        return;
    }

    // Slow path: rebuild entire chart
    svg.selectAll('*').remove();
    svg.attr('width', w).attr('height', h);

    // Title
    svg.append('text')
        .attr('class', 'chart-title')
        .attr('x', w / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .text(titles[mode]);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    questions.forEach((q, i) => {
        const val = values[i] || 0;
        const color = colors[i % colors.length];
        const yPos = y(i);

        // Background bar
        g.append('rect')
            .attr('class', 'bar-bg')
            .attr('x', 0)
            .attr('y', yPos)
            .attr('width', cw)
            .attr('height', y.bandwidth())
            .attr('rx', 4);

        // Value bar
        if (isDelta) {
            const barX = val >= 0 ? centerX : x(val);
            const barW = Math.abs(x(val) - centerX);
            g.append('rect')
                .attr('class', 'bar-fill')
                .attr('x', barX)
                .attr('y', yPos + 2)
                .attr('width', barW)
                .attr('height', y.bandwidth() - 4)
                .attr('rx', 3)
                .attr('fill', color);
        } else {
            g.append('rect')
                .attr('class', 'bar-fill')
                .attr('x', 0)
                .attr('y', yPos + 2)
                .attr('width', x(val))
                .attr('height', y.bandwidth() - 4)
                .attr('rx', 3)
                .attr('fill', color);
        }

        // Value label
        g.append('text')
            .attr('class', 'bar-value')
            .attr('x', cw - 5)
            .attr('y', yPos + y.bandwidth() / 2 + 4)
            .attr('text-anchor', 'end')
            .text((val >= 0 ? '+' : '') + val.toFixed(2));
    });

    // Center line for delta modes
    if (isDelta) {
        g.append('line')
            .attr('x1', centerX)
            .attr('x2', centerX)
            .attr('y1', 0)
            .attr('y2', ch)
            .attr('stroke', 'rgba(255,255,255,.3)')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '4,4');
    } else {
        // 0.5 reference line for core mode
        const midX = x(0.5);
        g.append('line')
            .attr('x1', midX)
            .attr('x2', midX)
            .attr('y1', 0)
            .attr('y2', ch)
            .attr('stroke', 'rgba(198,120,221,.5)')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '6,4');
        g.append('text')
            .attr('x', midX)
            .attr('y', -8)
            .attr('text-anchor', 'middle')
            .attr('fill', 'rgba(198,120,221,.8)')
            .attr('font-size', '10px')
            .attr('font-weight', '600')
            .text('0.5');
    }
}

function setupTrajLegend() {
    const leg = document.getElementById('trajExplorerLegend');
    leg.innerHTML = '';
    (dynamicsState.questions || []).forEach((q, i) => {
        const it = document.createElement('div');
        it.className = 'legend-item';
        it.innerHTML = `<div class="legend-swatch" style="background:${colors[i % colors.length]}"></div><span>${q}</span>`;
        leg.appendChild(it);
    });
}

// Add button to show trajectory explorer (add to view toggle)
function addTrajExplorerButton() {
    const viewToggle = document.getElementById('viewToggle');
    if (!viewToggle || document.getElementById('btnTimeline')) return;

    const btn = document.createElement('button');
    btn.className = 'mode-btn';
    btn.id = 'btnTimeline';
    btn.textContent = 'Timeline';
    btn.onclick = showTrajExplorer;
    viewToggle.appendChild(btn);
}

// Call when dynamics UI is shown
const origShowDynamicsUI = showDynamicsUI;
showDynamicsUI = function() {
    origShowDynamicsUI();
    addTrajExplorerButton();
};

// ════════════════════════════════════════════════════════════════════════════════
// CONVERGENCE MODE - Core values over samples (iterations)
// ════════════════════════════════════════════════════════════════════════════════
let showConvergence = false;
let convergenceState = {
    history: {},          // node_id -> list of running mean core values
    trajectories: {},     // node_id -> list of trajectory texts
    samples: {},          // node_id -> list of score vectors
    positions: [],        // sorted positions data
    currentPosIdx: 0,     // current position index (for slider)
    currentT: 0,          // float 0-1 for smooth slider
    isDragging: false,
    rafId: null,
    pendingT: null,
    selectedSample: null  // {posIdx, sampleIdx} when a dot is clicked
};

// Generate adaptive sample markers based on current sample count
function getConvergenceMarkers(maxSamples) {
    if (maxSamples < 3) return [];
    const markers = new Set();
    // Standard milestones
    [5, 10, 20, 50, 100, 200, 500, 1000].forEach(n => {
        if (n <= maxSamples && n >= 3) markers.add(n);
    });
    // Always include current max if notable
    if (maxSamples >= 5) markers.add(maxSamples);
    // Filter to avoid crowding - keep max 5 markers, well-spaced
    const sorted = [...markers].sort((a, b) => a - b);
    if (sorted.length <= 5) return sorted;
    // Keep first, last, and evenly spaced middle
    const result = [sorted[0]];
    const step = (sorted.length - 1) / 4;
    for (let i = 1; i < 4; i++) {
        result.push(sorted[Math.round(i * step)]);
    }
    result.push(sorted[sorted.length - 1]);
    return [...new Set(result)].sort((a, b) => a - b);
}

function showConvergenceMode() {
    showConvergence = true;
    showMagnitudes = false;
    document.getElementById('btnConvergence').classList.add('active');
    document.getElementById('btnEvolution').classList.remove('active');
    document.getElementById('btnMagnitudes').classList.remove('active');
    document.getElementById('evolutionModeToggle').classList.remove('visible');
    document.getElementById('diversityToggle').classList.remove('visible');
    document.getElementById('convergencePanel').classList.add('visible');
    setupConvergenceSlider();
    updateConvergenceSliderMax();
    setupDynamicsLegend();
    drawConvergenceChart();
}

function hideConvergenceMode() {
    showConvergence = false;
    document.getElementById('btnConvergence').classList.remove('active');
    document.getElementById('convergencePanel').classList.remove('visible');
    document.getElementById('trajectoriesPanel').classList.remove('visible');
    convergenceState.selectedSample = null;
}

document.getElementById('btnConvergence').onclick = () => {
    if (showConvergence) {
        hideConvergenceMode();
        document.getElementById('btnEvolution').click();
    } else {
        showConvergenceMode();
    }
};

// Update evolution/magnitudes buttons to hide convergence
const origEvolutionClick = document.getElementById('btnEvolution').onclick;
document.getElementById('btnEvolution').onclick = () => {
    hideConvergenceMode();
    origEvolutionClick();
};

const origMagnitudesClick = document.getElementById('btnMagnitudes').onclick;
document.getElementById('btnMagnitudes').onclick = () => {
    hideConvergenceMode();
    origMagnitudesClick();
};

function updateConvergenceData(d) {
    // Called from updateDynamicsPosition when new data arrives
    if (d.all_convergence) {
        convergenceState.history = d.all_convergence;
    }
    if (d.all_trajectories) {
        convergenceState.trajectories = d.all_trajectories;
    }
    if (d.all_samples) {
        convergenceState.samples = d.all_samples;
    }
    convergenceState.positions = dynamicsState.positions || [];
    updateConvergenceSliderMax();
    if (showConvergence) {
        drawConvergenceChart();
    }
}

function updateConvergenceSliderMax() {
    const numPos = convergenceState.positions.length;
    document.getElementById('convPositionMax').textContent = numPos > 0 ? numPos - 1 : 0;
}

// Global slider drag handler (shared by trajectory and convergence sliders)
let globalSliderDragState = null;

document.addEventListener('mousemove', (e) => {
    if (globalSliderDragState) {
        globalSliderDragState.update(e.clientX);
    }
});

document.addEventListener('mouseup', () => {
    if (globalSliderDragState) {
        globalSliderDragState.end();
        globalSliderDragState = null;
    }
});

function setupConvergenceSlider() {
    const track = document.getElementById('convSliderTrack');
    const ball = document.getElementById('convSliderBall');

    function updateFromMouse(clientX) {
        const rect = track.getBoundingClientRect();
        const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        const t = x / rect.width;
        setConvergencePositionSmooth(t);
    }

    track.onmousedown = (e) => {
        convergenceState.isDragging = true;
        globalSliderDragState = {
            update: updateFromMouse,
            end: () => { convergenceState.isDragging = false; }
        };
        updateFromMouse(e.clientX);
        e.preventDefault();
    };

    ball.style.left = '2px';
}

function setConvergencePositionSmooth(t) {
    t = Math.max(0, Math.min(1, t));
    convergenceState.pendingT = t;
    if (!convergenceState.rafId) {
        convergenceState.rafId = requestAnimationFrame(applyConvergencePosition);
    }
}

function applyConvergencePosition() {
    convergenceState.rafId = null;
    const t = convergenceState.pendingT;
    if (t === null) return;

    const numPos = convergenceState.positions.length;
    if (numPos === 0) return;

    convergenceState.currentT = t;
    convergenceState.currentPosIdx = Math.round(t * (numPos - 1));

    // Update slider ball
    const track = document.getElementById('convSliderTrack');
    const ball = document.getElementById('convSliderBall');
    const trackWidth = track.offsetWidth - ball.offsetWidth;
    ball.style.left = (2 + t * trackWidth) + 'px';

    // Update label
    document.getElementById('convPositionLabel').textContent = convergenceState.currentPosIdx;

    // Redraw chart
    drawConvergenceChart();
}

function drawConvergenceChart() {
    const w = innerWidth, h = innerHeight - 120;
    const m = { top: 100, right: 120, bottom: 100, left: 100 };
    d3.select('#canvas').selectAll('*').remove();
    const svg = d3.select('#canvas').attr('width', w).attr('height', h);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    const positions = convergenceState.positions;
    const history = convergenceState.history;
    const t = convergenceState.currentT;
    const questions = dynamicsState.questions || [];

    if (!positions.length || !questions.length) return;

    const numPos = positions.length;
    const floatIdx = t * (numPos - 1);
    const posIdx0 = Math.floor(floatIdx);
    const posIdx1 = Math.min(posIdx0 + 1, numPos - 1);
    const frac = floatIdx - posIdx0;

    // Get convergence histories for both positions
    // In dynamics analysis, node_id equals position index (set in build_dynamics_state)
    const history0 = history[posIdx0] || [];
    const history1 = history[posIdx1] || [];

    const cw = w - m.left - m.right;
    const ch = h - m.top - m.bottom;

    // Show placeholder if no samples yet
    if (history0.length === 0 && history1.length === 0) {
        g.append('text')
            .attr('x', cw / 2)
            .attr('y', ch / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#9a8baa')
            .attr('font-size', '14px')
            .text('Waiting for samples...');
        return;
    }

    const numSamples = Math.max(history0.length, history1.length);

    // Dynamically scale x-axis to actual sample count
    const xMax = Math.max(numSamples, 2);  // Minimum 2 for readable axis
    const x = d3.scaleLinear().domain([1, xMax]).range([0, cw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ch, 0]);

    // Grid lines
    g.append('g').attr('class', 'grid')
        .selectAll('line').data([0, 0.25, 0.5, 0.75, 1])
        .enter().append('line')
        .attr('x1', 0).attr('x2', cw)
        .attr('y1', d => y(d)).attr('y2', d => y(d));

    // Pre-compute interpolated values for all samples (outside question loop)
    // Convergence entries have {mean: [], std: []} structure
    const interpolatedSamples = [];
    for (let i = 0; i < numSamples; i++) {
        const e0 = history0[i];
        const e1 = history1[i];
        const s0 = e0?.mean;
        const s1 = e1?.mean;
        const std0 = e0?.std;
        const std1 = e1?.std;
        if (!s0 && !s1) continue;
        const interpolatedMean = [];
        const interpolatedStd = [];
        const nDims = Math.max(s0?.length || 0, s1?.length || 0);
        for (let d = 0; d < nDims; d++) {
            // Interpolate mean
            const v0 = s0?.[d];
            const v1 = s1?.[d];
            if (v0 !== undefined && v1 !== undefined) {
                interpolatedMean.push(v0 + (v1 - v0) * frac);
            } else if (v0 !== undefined) {
                interpolatedMean.push(v0);
            } else if (v1 !== undefined) {
                interpolatedMean.push(v1);
            } else {
                interpolatedMean.push(0);
            }
            // Interpolate std
            const sd0 = std0?.[d] ?? 0;
            const sd1 = std1?.[d] ?? 0;
            interpolatedStd.push(sd0 + (sd1 - sd0) * frac);
        }
        interpolatedSamples.push({ sampleNum: i + 1, mean: interpolatedMean, std: interpolatedStd });
    }

    // Draw line for each question dimension using pre-computed interpolated values
    questions.forEach((q, qIdx) => {
        if (highlightedQuestion !== null && highlightedQuestion !== qIdx) return;

        const color = colors[qIdx % colors.length];

        // Build line data from pre-computed interpolated samples
        // Clamp yLow/yHigh to [0, 1] because scores are normalized to this range
        const lineData = interpolatedSamples
            .filter(s => s.mean[qIdx] !== undefined)
            .map(s => ({
                x: s.sampleNum,
                y: s.mean[qIdx],
                yLow: Math.max(0, s.mean[qIdx] - s.std[qIdx]),
                yHigh: Math.min(1, s.mean[qIdx] + s.std[qIdx])
            }));

        if (lineData.length === 0) return;

        // Confidence band (±1 std) - draw first so it's behind the line
        // Only draw if we have multiple points and some variance exists
        if (lineData.length > 1 && lineData.some(d => d.yHigh > d.yLow)) {
            const area = d3.area()
                .x(d => x(d.x))
                .y0(d => y(d.yLow))
                .y1(d => y(d.yHigh))
                .curve(d3.curveLinear);

            g.append('path')
                .datum(lineData)
                .attr('d', area)
                .attr('fill', color)
                .attr('opacity', 0.15);
        }

        // Mean line
        const line = d3.line()
            .x(d => x(d.x))
            .y(d => y(d.y))
            .curve(d3.curveLinear);

        g.append('path')
            .datum(lineData)
            .attr('class', 'chart-line')
            .attr('d', line)
            .attr('stroke', color)
            .attr('stroke-width', 2.5);

        // Visible dots (drawn first so hit regions can reference them)
        const dots = g.selectAll(`.conv-dot-${qIdx}`)
            .data(lineData)
            .enter().append('circle')
            .attr('class', `chart-dot conv-dot-q${qIdx}`)
            .attr('cx', d => x(d.x))
            .attr('cy', d => y(d.y))
            .attr('r', 4)
            .attr('fill', color)
            .style('pointer-events', 'none');  // Let hit region handle clicks

        // Invisible vertical hit regions (larger click targets)
        g.selectAll(`.conv-hit-${qIdx}`)
            .data(lineData)
            .enter().append('rect')
            .attr('class', 'conv-hit-region')
            .attr('x', d => x(d.x) - 12)
            .attr('y', 0)
            .attr('width', 24)
            .attr('height', ch)
            .attr('fill', 'transparent')
            .style('cursor', 'pointer')
            .on('click', function(event, d) {
                const sampleIdx = d.x - 1;
                const posIdx = convergenceState.currentPosIdx;
                showConvergenceTrajectory(posIdx, sampleIdx);
            })
            .on('mouseover', function(event, d) {
                // Highlight all dots at this x position
                const sampleX = d.x;
                g.selectAll('.chart-dot')
                    .filter(dd => dd.x === sampleX)
                    .attr('r', 7)
                    .attr('stroke', 'white')
                    .attr('stroke-width', 2);
            })
            .on('mouseout', function() {
                // Reset all dots
                g.selectAll('.chart-dot')
                    .attr('r', 4)
                    .attr('stroke', 'none');
            });
    });

    // Draw selection highlight line if a sample is selected
    if (convergenceState.selectedSample) {
        const { posIdx: selPosIdx, sampleIdx: selSampleIdx } = convergenceState.selectedSample;
        // Only show if current position matches or is close to selected position
        if (Math.abs(posIdx0 - selPosIdx) <= 1 || Math.abs(posIdx1 - selPosIdx) <= 1) {
            const selSampleNum = selSampleIdx + 1;
            if (selSampleNum <= numSamples) {
                const xPos = x(selSampleNum);
                // Vertical line through chart
                g.append('line')
                    .attr('x1', xPos)
                    .attr('x2', xPos)
                    .attr('y1', 0)
                    .attr('y2', ch)
                    .attr('stroke', '#FF6B9D')
                    .attr('stroke-width', 2.5)
                    .attr('opacity', 0.8);
                // Label at top
                g.append('text')
                    .attr('x', xPos)
                    .attr('y', -8)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#FF6B9D')
                    .attr('font-size', '11px')
                    .attr('font-weight', '700')
                    .text(`#${selSampleNum}`);
            }
        }
    }

    // X-axis with bracket markers
    const xAxis = d3.axisBottom(x).ticks(Math.min(numSamples, 10));
    g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0,${ch})`)
        .call(xAxis);

    // Y-axis
    g.append('g')
        .attr('class', 'axis')
        .call(d3.axisLeft(y).ticks(5));

    // Add vertical reference lines and markers for key sample counts
    const markers = getConvergenceMarkers(numSamples);
    markers.forEach(n => {
        const xPos = x(n);
        // Vertical dotted line through the chart
        g.append('line')
            .attr('x1', xPos)
            .attr('x2', xPos)
            .attr('y1', 0)
            .attr('y2', ch)
            .attr('stroke', '#c490d1')
            .attr('stroke-width', 1.5)
            .attr('stroke-dasharray', '4,4')
            .attr('opacity', 0.6);
        // Circle marker at top of line
        g.append('circle')
            .attr('cx', xPos)
            .attr('cy', 0)
            .attr('r', 6)
            .attr('fill', '#c490d1')
            .attr('stroke', 'white')
            .attr('stroke-width', 2);
        // Label above
        g.append('text')
            .attr('x', xPos)
            .attr('y', -12)
            .attr('text-anchor', 'middle')
            .attr('fill', '#9b7cb8')
            .attr('font-size', '11px')
            .attr('font-weight', '700')
            .text(`n=${n}`);
    });

    // Axis labels
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -h / 2)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#6b7280')
        .text('Core Score');

    svg.append('text')
        .attr('x', w / 2)
        .attr('y', h - 25)
        .attr('text-anchor', 'middle')
        .attr('fill', '#6b7280')
        .text('Sample #');

    // Title with position info - interpolate label
    const pos0 = positions[posIdx0];
    const pos1 = positions[posIdx1];
    const posLabel = frac < 0.5 ? (pos0?.label || `Position ${posIdx0}`) : (pos1?.label || `Position ${posIdx1}`);
    const displaySamples = Math.max(history0.length, history1.length);
    svg.append('text')
        .attr('x', w / 2)
        .attr('y', 40)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4a3f5c')
        .attr('font-size', '16px')
        .attr('font-weight', '600')
        .attr('font-family', 'Roboto Mono, monospace')
        .text(`Convergence at "${posLabel}" (${displaySamples} samples)`);
}

function showConvergenceTrajectory(posIdx, sampleIdx) {
    // Get trajectory for the selected sample at current position
    const trajectories = convergenceState.trajectories[posIdx] || [];
    const samples = convergenceState.samples[posIdx] || [];
    const trajectory = trajectories[sampleIdx];
    const scores = samples[sampleIdx] || [];

    if (!trajectory) {
        showToast('warning', 'No Trajectory', 'Trajectory not available for this sample');
        return;
    }

    // Store selection
    convergenceState.selectedSample = { posIdx, sampleIdx };

    // Get position label
    const pos = convergenceState.positions[posIdx];
    const label = pos?.label || `Position ${posIdx}`;

    // Use the trajectoriesPanel (same as tree page)
    const panel = document.getElementById('trajectoriesPanel');
    const title = document.getElementById('trajectoriesTitle');
    const prefix = document.getElementById('trajectoriesPrefix');
    const list = document.getElementById('trajectoriesList');

    title.textContent = `Sample ${sampleIdx + 1} at "${label}"`;
    prefix.textContent = dynamicsState.prompt ? `Prompt: "${dynamicsState.prompt.slice(0, 80)}${dynamicsState.prompt.length > 80 ? '...' : ''}"` : '';

    // Build trajectory item
    list.innerHTML = '';
    const item = document.createElement('div');
    item.className = 'trajectory-item';

    // Find dominant color (highest score)
    if (scores.length) {
        let maxVal = Math.max(...scores);
        const EPSILON = 0.05;
        const topIndices = scores.map((v, i) => Math.abs(v - maxVal) <= EPSILON ? i : -1).filter(i => i >= 0);
        let dominantColor;
        if (topIndices.length === 1) {
            dominantColor = colors[topIndices[0] % colors.length];
        } else if (topIndices.length > 1) {
            dominantColor = blendColors(topIndices.map(j => colors[j % colors.length]));
        }
        if (dominantColor) {
            item.style.borderLeftColor = dominantColor;
            item.style.boxShadow = `inset 3px 0 0 ${dominantColor}`;
        }
    }

    const textDiv = document.createElement('div');
    // Show prefill dimmed, generated normal
    const prefill = dynamicsState.prefill || '';
    if (prefill && trajectory.startsWith(prefill)) {
        const prefillSpan = document.createElement('span');
        prefillSpan.className = 'trajectory-prefill';
        prefillSpan.textContent = prefill;
        const genSpan = document.createElement('span');
        genSpan.className = 'trajectory-generated';
        genSpan.textContent = trajectory.slice(prefill.length);
        textDiv.appendChild(prefillSpan);
        textDiv.appendChild(genSpan);
    } else {
        textDiv.textContent = trajectory;
    }
    item.appendChild(textDiv);

    // Show scores as badges
    if (scores.length) {
        const scoresDiv = document.createElement('div');
        scoresDiv.className = 'trajectory-scores';
        scores.forEach((score, j) => {
            const badge = document.createElement('span');
            badge.className = 'trajectory-score';
            badge.style.background = colors[j % colors.length];
            badge.textContent = score.toFixed(2);
            scoresDiv.appendChild(badge);
        });
        item.appendChild(scoresDiv);
    }

    list.appendChild(item);
    panel.classList.add('visible');

    // Redraw chart to show selection highlight
    drawConvergenceChart();
}

// Hook into position updates to track convergence data
const origUpdateDynamicsPosition = updateDynamicsPosition;
updateDynamicsPosition = function(d) {
    origUpdateDynamicsPosition(d);
    updateConvergenceData(d);
};
"""
