"""Dynamics page JavaScript: visualization and interaction."""

from __future__ import annotations


def get_dynamics_page_js() -> str:
    return """
// ════════════════════════════════════════════════════════════════════════════════
// DYNAMICS PAGE - Start & UI
// ════════════════════════════════════════════════════════════════════════════════
function startDynamics(){
    const keys=getApiKeys();
    // Check if API keys are needed (huggingface doesn't need them)
    const genNeedsKey = settings.gen_provider !== 'huggingface';
    const judgeNeedsKey = settings.judge_provider !== 'huggingface';
    const genKeyOk = !genNeedsKey || keys[settings.gen_provider];
    const judgeKeyOk = !judgeNeedsKey || keys[settings.judge_provider];
    if(!genKeyOk || !judgeKeyOk){alert('Configure API key in Settings');showSettings();return}
    currentMode='dynamics';
    const prompt = document.getElementById('dynPrompt').value;
    dynamicsState.prompt = prompt;  // Store prompt for trajectory explorer
    ws.send(JSON.stringify({
        action:'start_dynamics',
        api_keys:keys,
        prompt:prompt,
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
    document.getElementById('statValue1').textContent = dynamicsState.positions.length;
    document.getElementById('statValue2').textContent = dynamicsState.total_api_calls;
    document.getElementById('progressBar').style.width = ((d.progress || 0) * 100) + '%';

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
        // X-axis with word labels
        const xAxis = g.append('g').attr('class','axis').attr('transform',`translate(0,${ch})`).call(d3.axisBottom(x));
        xAxis.selectAll('text').attr('transform','rotate(-45)').attr('text-anchor','end').attr('dx','-0.5em').attr('dy','0.5em');
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

        // X-axis with word labels
        const xAxis = g.append('g').attr('class','axis').attr('transform',`translate(0,${ch})`).call(d3.axisBottom(x));
        xAxis.selectAll('text').attr('transform','rotate(-45)').attr('text-anchor','end').attr('dx','-0.5em').attr('dy','0.5em');
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
                drawDynamicsChart();
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

    // Initial chart
    updateTrajExplorer();

    // Reset mode buttons
    document.querySelectorAll('.traj-mode-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('trajModeCore').classList.add('active');

    // Add keyboard handler
    document.addEventListener('keydown', handleTrajKeydown);

    document.getElementById('trajExplorer').classList.add('visible');
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
        updateFromMouse(e.clientX);
        e.preventDefault();
    };

    document.addEventListener('mousemove', (e) => {
        if (trajExplorerState.isDragging) {
            updateFromMouse(e.clientX);
        }
    });

    document.addEventListener('mouseup', () => {
        if (trajExplorerState.isDragging) {
            trajExplorerState.isDragging = false;
            explorer.classList.remove('dragging');
        }
    });

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
"""
