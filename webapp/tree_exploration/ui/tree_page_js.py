"""Tree page JavaScript: visualization and interaction."""

from __future__ import annotations


def get_tree_page_js() -> str:
    return r"""
// ════════════════════════════════════════════════════════════════════════════════
// TREE PAGE - Data helpers
// ════════════════════════════════════════════════════════════════════════════════
function getNodeById(nodeId) {
    return treeState.nodes.find(n => n.node_id === nodeId);
}

function getNodeScores(nodeId) {
    const node = getNodeById(nodeId);
    return node ? node.core : [];
}

function getNodeOrientation(nodeId, refId) {
    // Get pre-computed expected relative orientation from server (list indexed by node_id)
    const node = getNodeById(nodeId);
    if (!node?.expected_relative_orientations) return null;
    const orientation = node.expected_relative_orientations[refId];
    return orientation?.length ? orientation : null;
}

// Pre-computed subtrees cache (computed once, used many times)
let subtreeCache = {};

function precomputeAllSubtrees() {
    // Compute subtree for every node upfront - O(n) total
    subtreeCache = {};

    if (!treeState?.nodes?.length) {
        console.warn('precomputeAllSubtrees called with no nodes');
        return;
    }

    const childrenOf = {};
    treeState.nodes.forEach(n => {
        childrenOf[n.node_id] = [];
    });
    treeState.nodes.forEach(n => {
        if (n.parent !== null && childrenOf[n.parent]) {
            childrenOf[n.parent].push(n.node_id);
        }
    });

    function buildSubtree(nodeId) {
        if (subtreeCache[nodeId]) return subtreeCache[nodeId];
        const set = new Set([nodeId]);
        (childrenOf[nodeId] || []).forEach(childId => {
            buildSubtree(childId).forEach(id => set.add(id));
        });
        subtreeCache[nodeId] = set;
        return set;
    }

    treeState.nodes.forEach(n => buildSubtree(n.node_id));
    console.log(`  ✓ Pre-computed ${Object.keys(subtreeCache).length} subtrees`);
}

function getSubtreeIds(nodeId) {
    // Handle undefined/null nodeId
    if (nodeId === undefined || nodeId === null) {
        return new Set([0]);  // Default to root
    }

    // Use cached version if available
    if (subtreeCache[nodeId]) return subtreeCache[nodeId];

    // Fallback to computation (shouldn't happen after init)
    if (!treeState?.nodes?.length) return new Set([nodeId]);

    const set = new Set([nodeId]);
    function addChildren(id) {
        treeState.nodes.filter(x => x.parent === id).forEach(c => {
            set.add(c.node_id);
            addChildren(c.node_id);
        });
    }
    addChildren(nodeId);
    return set;
}

// Compute normalized probabilities among siblings (from logprobs)
function computeSiblingProbabilities() {
    const probByNodeId = {};
    // Group nodes by parent
    const byParent = {};
    treeState.nodes.forEach(n => {
        const parentId = n.parent ?? 'root';
        if (!byParent[parentId]) byParent[parentId] = [];
        byParent[parentId].push(n);
    });
    // For each sibling group, normalize logprobs to probabilities
    Object.values(byParent).forEach(siblings => {
        const hasLogprobs = siblings.some(n => n.logprob !== undefined && n.logprob !== null);
        if (hasLogprobs) {
            // Convert logprobs to probs and normalize
            const probs = siblings.map(n => Math.exp(n.logprob || -10));
            const sum = probs.reduce((a, b) => a + b, 0);
            siblings.forEach((n, i) => {
                probByNodeId[n.node_id] = sum > 0 ? probs[i] / sum : 1 / siblings.length;
            });
        } else {
            // No logprobs - equal probability
            siblings.forEach(n => {
                probByNodeId[n.node_id] = 1 / siblings.length;
            });
        }
    });
    return probByNodeId;
}

function getNodeScale(nodeId, probabilities) {
    // Scale factor based on probability: min 0.6, max 1.2
    const prob = probabilities[nodeId] || 0.5;
    return 0.6 + prob * 0.6;
}

function getEdgeWidth(nodeId, probabilities) {
    // Edge width based on probability: min 1, max 4
    const prob = probabilities[nodeId] || 0.5;
    return 1 + prob * 3;
}

// ════════════════════════════════════════════════════════════════════════════════
// TREE PAGE - Start & UI
// ════════════════════════════════════════════════════════════════════════════════
function startTree(){
    const keys=getApiKeys();
    // Check if API keys are needed (huggingface doesn't need them)
    const genNeedsKey = settings.gen_provider !== 'huggingface';
    const judgeNeedsKey = settings.judge_provider !== 'huggingface';
    const genKeyOk = !genNeedsKey || keys[settings.gen_provider];
    const judgeKeyOk = !judgeNeedsKey || keys[settings.judge_provider];
    if(!genKeyOk || !judgeKeyOk){alert('Configure API key in Settings');showSettings();return}
    currentMode='tree';
    ws.send(JSON.stringify({
        action:'start_tree',
        api_keys:keys,
        prompt:document.getElementById('treePrompt').value,
        prefixes:getColoredEditorLines('treePrefixes'),
        questions:getColoredEditorLines('treeQuestions'),
        max_rounds:parseInt(document.getElementById('treeRounds').value),
        settings
    }));
    document.getElementById('treeConfig').classList.remove('visible');
    showTreeUI();
}

function showTreeUI(){
    console.log('🎨 showTreeUI called');
    ['floatingStats','modeToggle','controlButtons','legend','zoomControls'].forEach(id=>{
        const el = document.getElementById(id);
        if(el) {
            el.classList.add('visible');
            console.log(`  ✓ Made ${id} visible`);
        } else {
            console.error(`  ❌ Element ${id} not found!`);
        }
    });
    // Settings only accessible from landing page
}

// ════════════════════════════════════════════════════════════════════════════════
// TREE PAGE - Visualization
// ════════════════════════════════════════════════════════════════════════════════
let treeSvg,treeG,nodeGroups,zoomBehavior,currentTransform;
let referenceNodeId = 0;  // Default to root (node_id=0)
const margin={top:80,right:80,bottom:100,left:80};
const NODE_WIDTH = 280;  // Wider for text
const NODE_HEIGHT = 140; // Taller for text + bars
const TEXT_HEIGHT = 70;  // Space for text above bars
const BAR_AREA_HEIGHT = 60; // Space for bar chart

// Zoom controls
function zoomIn() { treeSvg.transition().duration(300).call(zoomBehavior.scaleBy, 1.3); }
function zoomOut() { treeSvg.transition().duration(300).call(zoomBehavior.scaleBy, 0.7); }
function zoomReset() { treeSvg.transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity); }
function zoomFit() {
    const bounds = treeG.node().getBBox();
    const fullWidth = treeSvg.node().clientWidth || innerWidth;
    const fullHeight = treeSvg.node().clientHeight || innerHeight;
    const scale = 0.9 * Math.min(fullWidth / bounds.width, fullHeight / bounds.height);
    const tx = (fullWidth - bounds.width * scale) / 2 - bounds.x * scale;
    const ty = (fullHeight - bounds.height * scale) / 2 - bounds.y * scale;
    treeSvg.transition().duration(500).call(zoomBehavior.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));
}

// Trajectories panel
let currentTrajectoriesNodeId = null;
let clickTimeout = null;  // For distinguishing single/double click

function showTrajectories(nodeId) {
    currentTrajectoriesNodeId = nodeId;
    const node = getNodeById(nodeId);
    if (!node) return;
    const panel = document.getElementById('trajectoriesPanel');
    const title = document.getElementById('trajectoriesTitle');
    const prefix = document.getElementById('trajectoriesPrefix');
    const list = document.getElementById('trajectoriesList');

    title.textContent = `Trajectories for "${(node.label || node.name).slice(0, 30)}"`;
    prefix.textContent = node.prefix ? `Prefix: "${node.prefix}"` : '(Root node)';

    list.innerHTML = '';
    const trajectories = node.trajectories || [];
    const samples = node.samples || [];
    if (trajectories.length === 0) {
        list.innerHTML = '<div style="color:var(--text-dim);padding:20px;text-align:center;font-style:italic">No trajectories yet. Sampling in progress...</div>';
    } else {
        trajectories.forEach((text, i) => {
            const item = document.createElement('div');
            item.className = 'trajectory-item';

            // Find dominant color for this trajectory (highest score)
            let dominantColor = null;
            if (samples[i] && samples[i].length) {
                const scores = samples[i];
                let maxVal = scores[0] || 0;
                scores.forEach((v) => { if (v > maxVal) maxVal = v; });
                // Find all indices within epsilon of max (for color blending)
                const EPSILON = 0.05;
                const topIndices = [];
                scores.forEach((v, j) => {
                    if (Math.abs(v - maxVal) <= EPSILON) topIndices.push(j);
                });
                if (topIndices.length === 1) {
                    dominantColor = colors[topIndices[0] % colors.length];
                } else {
                    dominantColor = blendColors(topIndices.map(j => colors[j % colors.length]));
                }
            }

            // Apply dominant color to border
            if (dominantColor) {
                item.style.borderLeftColor = dominantColor;
                item.style.boxShadow = `inset 3px 0 0 ${dominantColor}`;
            }

            const textDiv = document.createElement('div');
            textDiv.textContent = text;  // Show full text
            item.appendChild(textDiv);

            // Show scores for this trajectory
            if (samples[i]) {
                const scoresDiv = document.createElement('div');
                scoresDiv.className = 'trajectory-scores';
                samples[i].forEach((score, j) => {
                    const badge = document.createElement('span');
                    badge.className = 'trajectory-score';
                    badge.style.background = colors[j % colors.length];
                    badge.textContent = score.toFixed(2);
                    scoresDiv.appendChild(badge);
                });
                item.appendChild(scoresDiv);
            }
            list.appendChild(item);
        });
    }
    panel.classList.add('visible');
}

function hideTrajectories() {
    currentTrajectoriesNodeId = null;
    document.getElementById('trajectoriesPanel').classList.remove('visible');
}

function getUniqueText(nodeData, parentData) {
    // Show only the unique portion of this node's prefix (not shared with parent)
    const prefix = nodeData.prefix || '';
    const parentPrefix = parentData?.prefix || '';
    if (prefix.startsWith(parentPrefix)) {
        return prefix.slice(parentPrefix.length).trim();
    }
    return nodeData.label || nodeData.name || '';
}

function hexToRgb(hex) {
    // Parse hex color to RGB components
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (result) {
        return { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) };
    }
    // Handle rgb/rgba format
    const rgbMatch = hex.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (rgbMatch) {
        return { r: parseInt(rgbMatch[1]), g: parseInt(rgbMatch[2]), b: parseInt(rgbMatch[3]) };
    }
    return { r: 128, g: 128, b: 128 };
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => Math.round(x).toString(16).padStart(2, '0')).join('');
}

function blendColors(colorList) {
    // Blend multiple colors together
    if (!colorList.length) return '#808080';
    if (colorList.length === 1) return colorList[0];
    const rgbs = colorList.map(hexToRgb);
    const avg = {
        r: rgbs.reduce((s, c) => s + c.r, 0) / rgbs.length,
        g: rgbs.reduce((s, c) => s + c.g, 0) / rgbs.length,
        b: rgbs.reduce((s, c) => s + c.b, 0) / rgbs.length
    };
    return rgbToHex(avg.r, avg.g, avg.b);
}

function getDominantColor(nodeId) {
    // Get the color of the highest-scoring question for this node
    // In orientation mode, use absolute orientation values instead
    // If multiple scores tie for top, blend their colors
    const EPSILON = 0.05;  // Tolerance for considering scores "equal"

    if (showOrientation) {
        const orientation = getNodeOrientation(nodeId, referenceNodeId);
        if (!orientation || !orientation.length) return null;

        // Find max absolute value
        let maxVal = 0;
        orientation.forEach((v, i) => {
            const absV = Math.abs(v || 0);
            if (absV > maxVal) maxVal = absV;
        });

        // Find all indices within EPSILON of max
        const topIndices = [];
        orientation.forEach((v, i) => {
            if (Math.abs(Math.abs(v || 0) - maxVal) <= EPSILON) {
                topIndices.push(i);
            }
        });

        if (topIndices.length === 1) {
            return colors[topIndices[0] % colors.length];
        }
        return blendColors(topIndices.map(i => colors[i % colors.length]));
    } else {
        const scores = getNodeScores(nodeId);
        if (!scores || !scores.length) return null;

        // Find max value
        let maxVal = scores[0] || 0;
        scores.forEach((v, i) => { if (v > maxVal) maxVal = v; });

        // Find all indices within EPSILON of max
        const topIndices = [];
        scores.forEach((v, i) => {
            if (Math.abs(v - maxVal) <= EPSILON) {
                topIndices.push(i);
            }
        });

        if (topIndices.length === 1) {
            return colors[topIndices[0] % colors.length];
        }
        return blendColors(topIndices.map(i => colors[i % colors.length]));
    }
}

function updateNodeColors() {
    // Update text and card colors based on dominant score
    if (!nodeGroups) return;
    nodeGroups.each(function(d) {
        const isActualRoot = d.data.node_id === 0 && d.data.prefix === '';
        const isRef = d.data.node_id === referenceNodeId;
        const color = getDominantColor(d.data.node_id);
        // Only use pink for actual root with no scores; otherwise use dominant color
        const textColor = color || (isActualRoot ? '#c490d1' : '#4a3f5c');
        const cardStroke = isRef ? (color || '#7ec8c8') : 'rgba(180,160,200,.3)';
        d3.select(this).selectAll('.node-unique-text').attr('fill', textColor);
        d3.select(this).select('.node-card')
            .attr('stroke', cardStroke)
            .attr('stroke-width', isRef ? 2 : 1);
    });

    // Update link colors based on target node's dominant color
    const siblingProbs = computeSiblingProbabilities();
    const subtree = showOrientation ? getSubtreeIds(referenceNodeId) : null;
    treeG.selectAll('.link').each(function() {
        const link = d3.select(this);
        const targetId = parseInt(link.attr('data-target-id'));
        const inSubtree = !showOrientation || (subtree && subtree.has(targetId));
        const color = getDominantColor(targetId);
        const width = getEdgeWidth(targetId, siblingProbs);

        if (!inSubtree) {
            // Black & white: grayscale links for ancestors
            link.attr('stroke', 'rgba(90,90,90,0.25)')
                .attr('opacity', 0.35)
                .attr('stroke-width', width * 0.5);
        } else if (color) {
            link.attr('stroke', color).attr('opacity', 0.5).attr('stroke-width', width);
        } else {
            link.attr('stroke', 'rgba(100,120,150,0.3)').attr('opacity', 1).attr('stroke-width', width);
        }
    });

    // Update background color field
    updateColorField();
}

// Helper: calculate how many lines a node's text will wrap to
function getNodeTextLineCount(nodeData, parentData) {
    const prefix = nodeData.prefix || '';
    const parentPrefix = parentData?.prefix || '';
    let text = prefix.startsWith(parentPrefix) ? prefix.slice(parentPrefix.length).trim() : (nodeData.label || nodeData.name || '');
    const isRoot = nodeData.node_id === 0 || nodeData.name === 'root';
    if (isRoot) text = 'root';

    const maxCharsPerLine = 35;
    const words = text.split(' ');
    let lines = 1;
    let currentLen = 0;
    words.forEach(word => {
        if (currentLen + word.length + 1 <= maxCharsPerLine) {
            currentLen += word.length + 1;
        } else {
            lines++;
            currentLen = word.length;
        }
    });
    return Math.min(lines, 4);  // Cap at 4 lines max
}

function initTreeViz(){
    console.log('🌳 initTreeViz called');
    console.log('  treeState:', treeState);
    console.log('  treeState.nodes:', treeState?.nodes?.length, 'nodes');
    if (treeState?.nodes) {
        treeState.nodes.forEach(n => {
            console.log(`    [${n.node_id}] ${n.name}: prefix="${n.prefix?.slice(0,30)}..." parent=${n.parent}`);
        });
    }

    // Pre-compute all subtrees for fast orientation mode switching
    precomputeAllSubtrees();

    d3.select('#canvas').selectAll('*').remove();
    const tree=buildTreeData(),root=d3.hierarchy(tree);
    console.log('  Built tree hierarchy, descendants:', root.descendants().length);

    // Pre-calculate text line counts for each node
    const lineCountById = {};
    root.descendants().forEach(d => {
        lineCountById[d.data.node_id] = getNodeTextLineCount(d.data, d.parent?.data);
    });
    console.log('  Line counts:', lineCountById);

    // Count nodes at each depth to calculate required height
    const leafCount = root.leaves().length;
    const maxDepth = root.height;
    const totalNodes = root.descendants().length;

    // DYNAMIC spacing based on tree complexity
    const baseNodeSpacingY = NODE_HEIGHT + 140;
    const lineHeight = 22;
    const maxLines = Math.max(...Object.values(lineCountById), 1);
    const nodeSpacingY = baseNodeSpacingY + maxLines * lineHeight + 50;

    // Dynamic horizontal spacing: more depth = longer edges, more nodes = more space
    // Base of 300, scales with depth and total nodes
    const depthFactor = Math.max(1, maxDepth * 0.15);  // Deeper trees need more horizontal space
    const densityFactor = Math.max(1, Math.log10(totalNodes + 1) * 0.8);  // More nodes = more space
    const nodeSpacingX = NODE_WIDTH + Math.max(300, 250 * depthFactor * densityFactor);
    console.log(`  Dynamic spacing: depth=${maxDepth}, nodes=${totalNodes}, spacingX=${nodeSpacingX.toFixed(0)}`);

    const requiredHeight = Math.max(innerHeight, leafCount * nodeSpacingY + margin.top + margin.bottom);
    const requiredWidth = Math.max(innerWidth, (maxDepth + 1) * nodeSpacingX + margin.left + margin.right);

    treeSvg=d3.select('#canvas').attr('width',requiredWidth).attr('height',requiredHeight)
        .on('click', function(e) {
            // Close trajectories panel when clicking on background
            if (e.target.tagName === 'svg') hideTrajectories();
        });

    // Setup zoom behavior
    zoomBehavior = d3.zoom()
        .scaleExtent([0.1, 4])  // Min/max zoom levels
        .on('zoom', (event) => {
            currentTransform = event.transform;
            treeG.attr('transform', event.transform);
        });
    treeSvg.call(zoomBehavior);

    // Create main group for tree content
    treeG=treeSvg.append('g').attr('transform',`translate(${margin.left},${margin.top})`);

    // Use nodeSize with separation function - plenty of space to avoid collisions
    const layout=d3.tree()
        .nodeSize([baseNodeSpacingY, nodeSpacingX])
        .separation((a, b) => {
            // Generous vertical separation scaled by text lines
            const aLines = lineCountById[a.data.node_id] || 1;
            const bLines = lineCountById[b.data.node_id] || 1;
            const maxNodeLines = Math.max(aLines, bLines);
            // Base separation of 2.0, plus 0.6 per additional line
            return 2.0 + (maxNodeLines - 1) * 0.6;
        });
    layout(root);

    // Center the tree vertically
    const nodes = root.descendants();
    const minY = d3.min(nodes, d => d.x);
    const maxY = d3.max(nodes, d => d.x);
    const offsetY = (requiredHeight - margin.top - margin.bottom) / 2 - (minY + maxY) / 2;

    // Create gradient definitions for fading links
    const defs = treeSvg.append('defs');
    const linkGradient = defs.append('linearGradient')
        .attr('id', 'linkGradient')
        .attr('gradientUnits', 'userSpaceOnUse');
    linkGradient.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(100,120,150,0.5)');
    linkGradient.append('stop').attr('offset', '40%').attr('stop-color', 'rgba(100,120,150,0.3)');
    linkGradient.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(100,120,150,0.08)');

    // Compute sibling probabilities for sizing
    const siblingProbs = computeSiblingProbabilities();

    // Draw curved links with fade effect - thickness based on probability
    treeG.selectAll('.link').data(root.links()).enter().append('path')
        .attr('class','link')
        .attr('data-target-id', d => d.target.data.node_id)
        .attr('d',d3.linkHorizontal().x(d=>d.y).y(d=>d.x + offsetY))
        .attr('stroke', 'rgba(100,120,150,0.3)')
        .attr('stroke-width', d => getEdgeWidth(d.target.data.node_id, siblingProbs));

    // Create node groups with scale based on probability
    nodeGroups=treeG.selectAll('.node').data(nodes).enter().append('g')
        .attr('class','node')
        .attr('data-node-id',d=>d.data.node_id)
        .attr('data-scale', d => getNodeScale(d.data.node_id, siblingProbs))
        .attr('transform',d => {
            const scale = getNodeScale(d.data.node_id, siblingProbs);
            return `translate(${d.y},${d.x + offsetY}) scale(${scale})`;
        })
        .on('click',function(e,d){
            e.stopPropagation();
            const nodeEl = d3.select(this);

            // Immediate visual feedback - pulse effect
            nodeEl.select('.node-card')
                .transition().duration(60)
                .attr('transform', 'scale(1.08)')
                .transition().duration(60)
                .attr('transform', 'scale(1)');

            // Orientation mode: IMMEDIATE response, no timeout needed
            if (showOrientation) {
                setReferenceNode(d.data.node_id);
                return;
            }

            // Core mode: use timeout to distinguish single vs double click
            if (clickTimeout) {
                clearTimeout(clickTimeout);
                clickTimeout = null;
                // Double-click detected - set reference
                setReferenceNode(d.data.node_id);
            } else {
                clickTimeout = setTimeout(() => {
                    clickTimeout = null;
                    // Single click in core mode - show trajectories
                    showTrajectories(d.data.node_id);
                }, 180);
            }
        })
        .on('mouseenter', function(e,d) {
            const color = getDominantColor(d.data.node_id) || 'var(--pink)';
            d3.select(this).select('.node-card')
                .attr('filter', `drop-shadow(0 0 12px ${color})`);
            d3.select(this).selectAll('.node-unique-text')
                .attr('filter', `drop-shadow(0 0 6px ${color})`);
        })
        .on('mouseleave', function(e,d) {
            d3.select(this).select('.node-card').attr('filter', null);
            d3.select(this).selectAll('.node-unique-text').attr('filter', null);
        });

    // Large invisible hit area for easier clicking (50px padding around card)
    nodeGroups.append('rect')
        .attr('class','node-hit-area')
        .attr('x', -NODE_WIDTH/2 - 20)
        .attr('y', -75)
        .attr('width', NODE_WIDTH + 40)
        .attr('height', BAR_AREA_HEIGHT + 100)
        .attr('fill', 'transparent')
        .attr('cursor', 'pointer');

    // Background card for bars
    nodeGroups.append('rect')
        .attr('class','node-card')
        .attr('x', -NODE_WIDTH/2 + 30)
        .attr('y', -25)
        .attr('width', NODE_WIDTH - 60)
        .attr('height', BAR_AREA_HEIGHT)
        .attr('rx', 10);

    // Text displayed ABOVE the card - using SVG text with word wrapping
    nodeGroups.each(function(d) {
        const g = d3.select(this);
        const uniqueText = getUniqueText(d.data, d.parent?.data);
        const isRoot = d.data.node_id === 0 || d.data.name === 'root';
        const displayText = isRoot ? '⟨ root ⟩' : (uniqueText || d.data.label || '…');

        // Wrap text into lines (simple word wrap)
        const maxCharsPerLine = 35;
        const words = displayText.split(' ');
        const lines = [];
        let currentLine = '';
        words.forEach(word => {
            if ((currentLine + ' ' + word).trim().length <= maxCharsPerLine) {
                currentLine = (currentLine + ' ' + word).trim();
            } else {
                if (currentLine) lines.push(currentLine);
                currentLine = word;
            }
        });
        if (currentLine) lines.push(currentLine);

        // Limit to 3 lines max
        if (lines.length > 3) {
            lines.length = 3;
            lines[2] = lines[2].slice(0, -3) + '...';
        }

        const textGroup = g.append('g').attr('class', 'node-text-group');
        const lineHeight = 16;
        const startY = -35 - (lines.length * lineHeight);

        lines.forEach((line, i) => {
            textGroup.append('text')
                .attr('class', 'node-unique-text')
                .attr('x', 0)
                .attr('y', startY + i * lineHeight)
                .attr('text-anchor', 'middle')
                .attr('fill', isRoot ? '#c490d1' : '#4a3f5c')
                .attr('font-size', isRoot ? '12px' : '13px')
                .attr('font-weight', '600')
                .attr('font-style', isRoot ? 'italic' : 'normal')
                .text(line);
        });

    });

    referenceNodeId = 0;  // Reset to root
    drawTreeBars();
    updateNodeColors();
    drawColorField(nodes, offsetY, defs);
    setupTreeLegend();
    updateTreeStats();
    drawPhantomBranches(nodes, offsetY, defs);
}

function drawColorField(nodes, offsetY, defs) {
    // Create color field layer - will be repositioned after phantom branches are drawn
    const fieldG = treeG.append('g').attr('class', 'color-field');

    nodes.forEach(d => {
        const x = d.y;
        const y = d.x + offsetY;
        const nodeId = d.data.node_id;

        // Create radial gradient for this node's glow - intense and localized for vector field
        const gradId = `colorField-${nodeId}`;
        const grad = defs.append('radialGradient')
            .attr('id', gradId)
            .attr('cx', '50%').attr('cy', '50%')
            .attr('r', '50%');
        // Very bright center that fades out sharply
        grad.append('stop')
            .attr('class', `field-stop-${nodeId}`)
            .attr('offset', '0%')
            .attr('stop-color', 'rgba(100,120,150,0.6)');
        grad.append('stop')
            .attr('class', `field-stop-mid-${nodeId}`)
            .attr('offset', '30%')
            .attr('stop-color', 'rgba(100,120,150,0.35)');
        grad.append('stop')
            .attr('class', `field-stop-outer-${nodeId}`)
            .attr('offset', '60%')
            .attr('stop-color', 'rgba(100,120,150,0.12)');
        grad.append('stop')
            .attr('offset', '100%')
            .attr('stop-color', 'rgba(0,0,0,0)');

        // Glow ellipse - smaller radius for more localized effect
        fieldG.append('ellipse')
            .attr('class', `field-glow field-glow-${nodeId}`)
            .attr('cx', x)
            .attr('cy', y)
            .attr('rx', 140)
            .attr('ry', 90)
            .attr('fill', `url(#${gradId})`);
    });
}

function updateColorField() {
    // Update color field to match current node colors - vector field effect
    // Orientation styling is handled by CSS classes (applied in applyOrientationStyling)
    if (!treeG || !treeState?.nodes) return;

    treeState.nodes.forEach(n => {
        const color = getDominantColor(n.node_id);
        if (color) {
            // Set the color - CSS handles opacity for ancestors via .orient-ancestor class
            d3.select(`.field-stop-${n.node_id}`)
                .attr('stop-color', color)
                .attr('stop-opacity', 0.65);
            d3.select(`.field-stop-mid-${n.node_id}`)
                .attr('stop-color', color)
                .attr('stop-opacity', 0.4);
            d3.select(`.field-stop-outer-${n.node_id}`)
                .attr('stop-color', color)
                .attr('stop-opacity', 0.15);
        }
    });
}

function drawPhantomBranches(nodes, offsetY, defs) {
    // Add phantom/ghost branches - proper DIVERGING tree structure (trees don't converge!)
    // Lower to be BEHIND color field (color field overlays on top)
    const phantomG = treeG.append('g').attr('class', 'phantom-branches');
    phantomG.lower();  // Put at very back
    // Now move color-field to be just above phantom-branches
    const colorField = treeG.select('.color-field');
    if (!colorField.empty()) {
        colorField.raise();  // Bring color field above phantom branches
        // But keep it below links and nodes
        treeG.selectAll('.link').raise();
        treeG.selectAll('.node').raise();
    }

    // Find leaf nodes (no children)
    const leafNodes = nodes.filter(d => !nodes.some(n => n.parent && n.parent.data.node_id === d.data.node_id));

    // Phantom node dimensions
    const PHANTOM_WIDTH = 70;
    const PHANTOM_HEIGHT = 26;

    // Helper: create a fading gradient - pastel lavender tones, subtle
    function makeGradient(id, x1, y1, x2, y2, startOpacity, endOpacity) {
        const grad = defs.append('linearGradient')
            .attr('id', id)
            .attr('gradientUnits', 'userSpaceOnUse')
            .attr('x1', x1).attr('y1', y1)
            .attr('x2', x2).attr('y2', y2);
        // Subtle pastel lavender - more transparent
        grad.append('stop').attr('offset', '0%').attr('stop-color', `rgba(180,160,200,${startOpacity * 0.2})`);
        grad.append('stop').attr('offset', '100%').attr('stop-color', `rgba(160,140,180,${endOpacity * 0.1})`);
        return id;
    }

    // Helper: draw a phantom node - soft pastel, subtle
    function drawPhantomNode(x, y, opacity, scale = 1) {
        const w = PHANTOM_WIDTH * scale;
        const h = PHANTOM_HEIGHT * scale;
        // Subtle - multiply opacity by 0.15
        const fadeOp = opacity * 0.15;
        phantomG.append('rect')
            .attr('x', x - w/2)
            .attr('y', y - h/2)
            .attr('width', w)
            .attr('height', h)
            .attr('rx', 5)
            .attr('fill', `rgba(248,244,255,${fadeOp * 0.3})`)
            .attr('stroke', `rgba(180,160,200,${fadeOp * 0.2})`)
            .attr('stroke-width', 0.5)
            .attr('stroke-dasharray', '2,3');  // Dashed for artsy effect
        // Bar placeholder - pastel colored
        phantomG.append('rect')
            .attr('x', x - 15 * scale)
            .attr('y', y - 3 * scale)
            .attr('width', 30 * scale)
            .attr('height', 6 * scale)
            .attr('rx', 2)
            .attr('fill', `rgba(196,144,209,${fadeOp * 0.1})`);
    }

    // Helper: draw diverging phantom subtree recursively
    function drawPhantomSubtree(startX, startY, depth, maxDepth, baseOpacity, spreadAngle) {
        if (depth > maxDepth) return;

        const opacity = baseOpacity * Math.pow(0.7, depth);
        const branchLen = 90 + Math.random() * 50 - depth * 12;
        const numBranches = depth === 0 ? (2 + Math.floor(Math.random() * 3)) : (Math.random() > 0.3 ? 2 : 1);

        for (let i = 0; i < numBranches; i++) {
            // Spread branches out more (diverging, not converging)
            const angleOffset = (i - (numBranches - 1) / 2) * spreadAngle;
            const endX = startX + branchLen;
            const endY = startY + angleOffset * (50 + Math.random() * 30);

            // Draw branch line - more visible
            const gradId = `phantomTree-${Math.random().toString(36).slice(2,8)}`;
            makeGradient(gradId, startX, startY, endX, endY, opacity * 0.6, opacity * 0.2);

            phantomG.append('path')
                .attr('d', `M${startX},${startY} Q${startX + branchLen * 0.6},${startY + angleOffset * 18} ${endX},${endY}`)
                .attr('fill', 'none')
                .attr('stroke', `url(#${gradId})`)
                .attr('stroke-width', Math.max(1.2, 2.5 - depth * 0.4));

            // Draw phantom node at end
            drawPhantomNode(endX + PHANTOM_WIDTH/2 + 5, endY, opacity, Math.max(0.6, 1 - depth * 0.12));

            // Recurse for deeper branches - more likely to continue
            if (depth < maxDepth && Math.random() > 0.2) {
                drawPhantomSubtree(endX + PHANTOM_WIDTH + 10, endY, depth + 1, maxDepth, baseOpacity, spreadAngle * 0.75);
            }
        }
    }

    // Draw phantom subtrees extending from each leaf node
    leafNodes.forEach(d => {
        const x = d.y + 80;  // Start just past the leaf node
        const y = d.x + offsetY;

        // Draw 2-4 levels of diverging phantom branches
        const maxDepth = 2 + Math.floor(Math.random() * 2);  // 2-3 levels deep
        drawPhantomSubtree(x, y, 0, maxDepth, 0.9, 1.0 + Math.random() * 0.5);
    });

    // Also add phantom branches diverging from internal nodes (showing unexplored paths)
    const internalNodes = nodes.filter(d =>
        nodes.some(n => n.parent && n.parent.data.node_id === d.data.node_id) && d.data.parent !== null
    );
    internalNodes.forEach(d => {
        // Only sometimes add phantom branches to internal nodes
        if (Math.random() > 0.5) return;

        const x = d.y + 80;
        const y = d.x + offsetY;

        // Get existing children y positions to avoid overlap
        const childYs = nodes
            .filter(n => n.parent && n.parent.data.node_id === d.data.node_id)
            .map(n => n.x + offsetY);

        // Add 1-2 phantom "unexplored" branches
        const numPhantom = 1 + Math.floor(Math.random() * 2);
        for (let i = 0; i < numPhantom; i++) {
            // Pick a y position that doesn't overlap with real children
            let phantomY = y + (Math.random() - 0.5) * 120;
            // Make sure it's not too close to existing children
            const tooClose = childYs.some(cy => Math.abs(cy - phantomY) < 40);
            if (tooClose) phantomY = y + (Math.random() > 0.5 ? 1 : -1) * (60 + Math.random() * 40);

            const branchLen = 60 + Math.random() * 40;
            const endX = x + branchLen;
            const endY = phantomY;

            const gradId = `phantomInternal-${Math.random().toString(36).slice(2,8)}`;
            makeGradient(gradId, x, y, endX, endY, 0.18, 0.05);

            phantomG.append('path')
                .attr('d', `M${x},${y} Q${x + branchLen * 0.5},${(y + endY) / 2} ${endX},${endY}`)
                .attr('fill', 'none')
                .attr('stroke', `url(#${gradId})`)
                .attr('stroke-width', 1.5);

            drawPhantomNode(endX + 40, endY, 0.4, 0.8);
        }
    });

    // Add ACTUAL PROMPT text leading INTO root - VERY PROMINENT (5x bigger, much further)
    const rootNode = nodes.find(d => d.data.parent === null);
    if (rootNode) {
        const rootX = rootNode.y;
        const rootY = rootNode.x + offsetY;

        // Get the actual prompt text
        const promptText = document.getElementById('treePrompt')?.value || 'Prompt';

        // Wrap prompt text into lines (wider lines for bigger box)
        const maxCharsPerLine = 55;
        const promptWords = promptText.trim().split(/\s+/);
        const promptLines = [];
        let currentLine = '';
        promptWords.forEach(word => {
            if ((currentLine + ' ' + word).trim().length <= maxCharsPerLine) {
                currentLine = (currentLine + ' ' + word).trim();
            } else {
                if (currentLine) promptLines.push(currentLine);
                currentLine = word;
            }
        });
        if (currentLine) promptLines.push(currentLine);

        // Limit to 8 lines for bigger box
        if (promptLines.length > 8) {
            promptLines.length = 8;
            promptLines[7] = promptLines[7].slice(0, -3) + '...';
        }

        // Prompt box - 5X BIGGER and 3X FURTHER away
        const lineHeight = 42;
        const boxHeight = Math.max(200, promptLines.length * lineHeight + 100);
        const boxWidth = 700;
        const promptX = rootX - 2700;  // 3x further left
        const promptY = rootY;

        // Create prompt group for orientation greying
        const promptGroup = phantomG.append('g').attr('class', 'prompt-box-group');

        // Draw prompt box with NEUTRAL gradient background
        const promptBoxGrad = defs.append('linearGradient')
            .attr('id', 'promptBoxGrad')
            .attr('x1', '0%').attr('y1', '0%')
            .attr('x2', '100%').attr('y2', '100%');
        promptBoxGrad.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(45,50,60,0.9)');
        promptBoxGrad.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(35,40,50,0.9)');

        promptGroup.append('rect')
            .attr('x', promptX - boxWidth/2)
            .attr('y', promptY - boxHeight/2)
            .attr('width', boxWidth)
            .attr('height', boxHeight)
            .attr('rx', 24)
            .attr('fill', 'url(#promptBoxGrad)')
            .attr('stroke', 'rgba(140,150,170,0.5)')  // Neutral border
            .attr('stroke-width', 3)
            .attr('filter', 'drop-shadow(0 8px 25px rgba(0,0,0,0.4))');

        // "PROMPT" label - NEUTRAL color
        promptGroup.append('text')
            .attr('x', promptX)
            .attr('y', promptY - boxHeight/2 + 45)
            .attr('text-anchor', 'middle')
            .attr('fill', 'rgba(160,170,190,0.85)')  // Neutral grey-blue
            .attr('font-size', '28px')
            .attr('font-weight', '700')
            .attr('letter-spacing', '0.2em')
            .text('PROMPT');

        // Draw actual prompt text lines - NEUTRAL
        const textStartY = promptY - boxHeight/2 + 95;
        promptLines.forEach((line, i) => {
            promptGroup.append('text')
                .attr('x', promptX)
                .attr('y', textStartY + i * lineHeight)
                .attr('text-anchor', 'middle')
                .attr('fill', 'rgba(200,205,215,0.9)')  // Neutral text
                .attr('font-size', '28px')
                .attr('font-weight', '500')
                .attr('font-style', 'italic')
                .text(line);
        });

        // Draw LONG curved edge from Prompt to root - NEUTRAL colors
        const gradPromptId = 'phantomPromptGrad';
        const gradPrompt = defs.append('linearGradient')
            .attr('id', gradPromptId)
            .attr('gradientUnits', 'userSpaceOnUse')
            .attr('x1', promptX + boxWidth/2).attr('y1', promptY)
            .attr('x2', rootX - 80).attr('y2', rootY);
        gradPrompt.append('stop').attr('offset', '0%').attr('stop-color', 'rgba(140,150,170,0.6)');
        gradPrompt.append('stop').attr('offset', '30%').attr('stop-color', 'rgba(130,140,160,0.4)');
        gradPrompt.append('stop').attr('offset', '70%').attr('stop-color', 'rgba(120,130,150,0.5)');
        gradPrompt.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(110,120,140,0.7)');

        // LONG curved path with elegant S-curve bezier
        const startX = promptX + boxWidth/2;
        const endX = rootX - 80;
        const midX1 = startX + (endX - startX) * 0.4;
        const midX2 = startX + (endX - startX) * 0.6;

        promptGroup.append('path')
            .attr('d', `M${startX},${promptY} C${midX1},${promptY} ${midX2},${rootY} ${endX},${rootY}`)
            .attr('fill', 'none')
            .attr('stroke', `url(#${gradPromptId})`)
            .attr('stroke-width', 5)
            .attr('stroke-linecap', 'round');

        // Add arrow at end pointing to root
        promptGroup.append('path')
            .attr('d', `M${endX - 18},${rootY - 12} L${endX},${rootY} L${endX - 18},${rootY + 12}`)
            .attr('fill', 'none')
            .attr('stroke', 'rgba(110,120,140,0.7)')
            .attr('stroke-width', 3)
            .attr('stroke-linecap', 'round')
            .attr('stroke-linejoin', 'round');
    }

    // Add some ambient floating dots/particles - very subtle
    const bounds = treeG.node().getBBox();
    for (let i = 0; i < 40; i++) {
        const px = bounds.x + Math.random() * (bounds.width + 400);
        const py = bounds.y - 100 + Math.random() * (bounds.height + 200);
        phantomG.append('circle')
            .attr('cx', px)
            .attr('cy', py)
            .attr('r', 1 + Math.random() * 2.5)
            .attr('fill', `rgba(180,160,200,${0.03 + Math.random() * 0.04})`)
            .attr('class', 'phantom-particle');
    }

    // Add scattered faint mini-trees in background to show "many possible subtrees"
    drawScatteredMiniTrees(phantomG, bounds, defs);
}

// Draw faint mini-tree diagrams scattered in the background
function drawScatteredMiniTrees(phantomG, bounds, defs) {
    const numTrees = 12 + Math.floor(Math.random() * 8);  // 12-20 mini trees

    for (let t = 0; t < numTrees; t++) {
        // Random position - spread widely around and beyond the main tree
        const treeX = bounds.x - 400 + Math.random() * (bounds.width + 800);
        const treeY = bounds.y - 200 + Math.random() * (bounds.height + 400);

        // Very faint opacity - more subtle
        const baseOp = 0.02 + Math.random() * 0.03;

        // Draw a simple mini-tree structure
        drawMiniTree(phantomG, treeX, treeY, baseOp, defs);
    }
}

function drawMiniTree(g, startX, startY, opacity, defs) {
    const branchLen = 25 + Math.random() * 20;
    const nodeSize = 8 + Math.random() * 6;
    const levels = 2 + Math.floor(Math.random() * 2);  // 2-3 levels

    // Draw root node - subtle
    g.append('rect')
        .attr('x', startX - nodeSize/2)
        .attr('y', startY - nodeSize/2)
        .attr('width', nodeSize)
        .attr('height', nodeSize * 0.7)
        .attr('rx', 2)
        .attr('fill', `rgba(248,244,255,${opacity * 0.3})`)
        .attr('stroke', `rgba(180,160,200,${opacity * 0.2})`)
        .attr('stroke-width', 0.5);

    // Recursively draw branches
    drawMiniTreeBranches(g, startX + nodeSize/2, startY, 0, levels, branchLen, nodeSize, opacity);
}

function drawMiniTreeBranches(g, x, y, depth, maxDepth, branchLen, nodeSize, opacity) {
    if (depth >= maxDepth) return;

    const numBranches = depth === 0 ? (2 + Math.floor(Math.random() * 2)) : (1 + Math.floor(Math.random() * 2));
    const spreadY = 20 + Math.random() * 15;
    const fadeOp = opacity * Math.pow(0.7, depth);

    for (let i = 0; i < numBranches; i++) {
        const offsetY = (i - (numBranches - 1) / 2) * spreadY;
        const endX = x + branchLen * (0.8 + Math.random() * 0.4);
        const endY = y + offsetY;

        // Draw branch line - subtle
        g.append('path')
            .attr('d', `M${x},${y} Q${x + branchLen * 0.5},${(y + endY) / 2} ${endX},${endY}`)
            .attr('fill', 'none')
            .attr('stroke', `rgba(180,160,200,${fadeOp * 0.25})`)
            .attr('stroke-width', 0.8);

        // Draw node - subtle
        const ns = nodeSize * (0.8 - depth * 0.15);
        g.append('rect')
            .attr('x', endX)
            .attr('y', endY - ns * 0.35)
            .attr('width', ns)
            .attr('height', ns * 0.7)
            .attr('rx', 2)
            .attr('fill', `rgba(248,244,255,${fadeOp * 0.2})`)
            .attr('stroke', `rgba(180,160,200,${fadeOp * 0.15})`)
            .attr('stroke-width', 0.5);

        // Recurse
        if (Math.random() > 0.3) {
            drawMiniTreeBranches(g, endX + ns, endY, depth + 1, maxDepth, branchLen * 0.8, nodeSize, opacity);
        }
    }
}

function buildTreeData(){
    console.log('🔨 buildTreeData called');
    console.log('  treeState.nodes:', treeState.nodes);

    const m={};
    treeState.nodes.forEach(n=>{
        m[n.node_id]={...n,children:[]};
        console.log(`  Created map entry [${n.node_id}]: parent=${n.parent}`);
    });

    let root=null;
    treeState.nodes.forEach(n=>{
        if(n.parent!==null&&m[n.parent]!==undefined){
            m[n.parent].children.push(m[n.node_id]);
            console.log(`  Linked [${n.node_id}] to parent [${n.parent}]`);
        } else {
            root=m[n.node_id];
            console.log(`  Set root to [${n.node_id}]`);
        }
    });

    console.log('  Final tree root:', root);
    return root||{node_id:0,name:'root',children:[]};
}

// Reference node update - immediate response, debounce expensive operations
let referenceUpdatePending = null;

function setReferenceNode(nodeId){
    referenceNodeId=nodeId;

    // Immediate visual feedback - just update CSS classes (fast)
    applyOrientationStyling();

    // Cancel pending expensive update and schedule new one
    if (referenceUpdatePending) {
        cancelAnimationFrame(referenceUpdatePending);
    }

    // Debounce expensive operations (bars, colors, phantom)
    referenceUpdatePending = requestAnimationFrame(() => {
        drawTreeBars();
        updateNodeColors();
        updatePhantomOrientationMode();
        referenceUpdatePending = null;
    });
}

// Cache for subtree IDs to avoid recomputation
let cachedSubtreeIds = null;
let cachedSubtreeRefNode = null;

function applyOrientationStyling() {
    // Fast CSS class-based styling
    if (!nodeGroups || !treeG) return;

    // Use cached subtree if reference node hasn't changed
    if (showOrientation) {
        if (cachedSubtreeRefNode !== referenceNodeId) {
            cachedSubtreeIds = getSubtreeIds(referenceNodeId);
            cachedSubtreeRefNode = referenceNodeId;
        }
    }
    const subtree = showOrientation ? cachedSubtreeIds : null;

    // Single pass: update data attribute on SVG for CSS-based styling
    const svgEl = document.getElementById('canvas');
    if (svgEl) {
        svgEl.dataset.orientRef = showOrientation ? referenceNodeId : '';
    }

    // Batch class updates using native DOM for speed
    if (subtree) {
        nodeGroups.each(function(d) {
            const isAncestor = !subtree.has(d.data.node_id);
            this.classList.toggle('orient-ancestor', isAncestor);
        });
    } else {
        nodeGroups.classed('orient-ancestor', false);
    }

    // Prompt box
    const promptBox = treeG.select('.prompt-box-group').node();
    if (promptBox) promptBox.classList.toggle('orient-ancestor', showOrientation);
}

function drawTreeBars(){
    if (!nodeGroups) return;  // Guard against early calls

    const bw = (NODE_WIDTH/2) - 40;  // Bar max width
    const subtree = showOrientation ? cachedSubtreeIds : null;
    nodeGroups.each(function(d){
        const g=d3.select(this);
        const scores=getNodeScores(d.data.node_id);
        if(!scores.length){
            g.selectAll('.bar-group').remove();
            return;
        }

        const bh=Math.min(8, (BAR_AREA_HEIGHT - 20)/scores.length);
        const sy=-((scores.length*bh)/2);
        const inSubtree = !showOrientation || (subtree && subtree.has(d.data.node_id));
        const expectedRelativeOrientation = showOrientation ? getNodeOrientation(d.data.node_id, referenceNodeId) : null;

        // Check if we can update existing bars instead of recreating
        let bg = g.select('.bar-group');
        if (bg.empty()) {
            bg = g.append('g').attr('class','bar-group').attr('transform', 'translate(0, 5)');
            // Center line
            bg.append('line').attr('class','center-line')
                .attr('x1',0).attr('y1',sy-3)
                .attr('x2',0).attr('y2',sy+scores.length*bh+3)
                .attr('stroke','rgba(100,120,150,.4)')
                .attr('stroke-width', 1);
            // Create bars
            scores.forEach((v,i)=>{
                bg.append('rect')
                    .attr('class','bar bar-'+i)
                    .attr('y',sy+i*bh+1)
                    .attr('height',bh-2)
                    .attr('fill',colors[i%colors.length])
                    .attr('rx',3);
            });
        }

        // Update bar positions/opacities (fast - no DOM creation)
        scores.forEach((v,i)=>{
            const relOrientation = expectedRelativeOrientation ? expectedRelativeOrientation[i] : null;
            const geom = CoreMath.computeBarGeometry(v, relOrientation, bw, showOrientation);
            const hi=highlightedQuestion===null||highlightedQuestion===i;
            const baseOpacity = inSubtree ? (hi ? 0.9 : 0.2) : 0.15;
            bg.select('.bar-'+i)
                .attr('x',geom.x)
                .attr('width',geom.width)
                .attr('opacity',baseOpacity);
        });
    });
}

// Track phantom canvases per node - ONE canvas per node for all effects
const nodePhantomCanvases = {};
const nodePhantomData = {};  // Store trajectory data for redrawing
const nodePhantomDrawPending = {};  // Debounce redraws for performance
const MAX_PHANTOM_TRAJECTORIES_PER_NODE = 15;  // Limit to prevent lag
let phantomDrawThrottleMs = 150;  // Throttle phantom drawing

function isMiddleNode(nodeId) {
    // Check if this node has children (is a middle/internal node, not a leaf)
    return treeState.nodes.some(n => n.parent === nodeId);
}

function getTopScoringColors(scores) {
    // Get colors for all top-scoring questions (within epsilon of max)
    if (!scores || !scores.length) return [{ r: 100, g: 120, b: 150 }];

    const EPSILON = 0.08;
    let maxVal = Math.max(...scores);
    const topIndices = [];
    scores.forEach((v, i) => {
        if (Math.abs(v - maxVal) <= EPSILON) topIndices.push(i);
    });

    return topIndices.map(i => {
        const rgb = hexToRgb(colors[i % colors.length]);
        return { r: rgb.r, g: rgb.g, b: rgb.b, idx: i };
    });
}

function addDynamicPhantomBranch(nodeId, trajectoryText, scores) {
    // OPTIMIZED: All phantom effects drawn on ONE canvas per node
    // Guard against early calls before tree is initialized
    if (!treeG || !nodeGroups) return;

    const nodeGroup = nodeGroups.filter(x => x.data.node_id === nodeId);
    if (nodeGroup.empty()) return;

    const transform = nodeGroup.attr('transform');
    const match = transform.match(/translate\(([^,]+),([^)]+)\)/);
    if (!match) return;

    const nodeX = parseFloat(match[1]);
    const nodeY = parseFloat(match[2]);

    // Extract words from trajectory
    const node = getNodeById(nodeId);
    const prefix = node?.prefix || '';
    let snippet = trajectoryText;
    if (snippet.startsWith(prefix)) {
        snippet = snippet.slice(prefix.length).trim();
    }
    const allWords = snippet.split(/\s+/).filter(w => w.length > 0);
    if (allWords.length === 0) return;

    // Get ALL top-scoring colors for this trajectory (for staggering)
    const topColors = getTopScoringColors(scores);

    // Check if this is a middle node (has children)
    const isMiddle = isMiddleNode(nodeId);

    // Store trajectory data for this node - include all top colors for staggering
    // Limit stored trajectories to prevent memory/performance issues
    if (!nodePhantomData[nodeId]) nodePhantomData[nodeId] = [];
    nodePhantomData[nodeId].push({ words: allWords, topColors, nodeX, nodeY, isMiddle });
    if (nodePhantomData[nodeId].length > MAX_PHANTOM_TRAJECTORIES_PER_NODE) {
        nodePhantomData[nodeId].shift();  // Remove oldest
    }

    // Create or get the foreignObject + canvas for this node
    const phantomG = treeG.select('.phantom-branches');
    if (phantomG.empty()) return;

    // Middle nodes: shorter, more vertical canvas to avoid overlapping with children
    const canvasWidth = isMiddle ? 350 : 800;
    const canvasHeight = isMiddle ? 600 : 500;

    if (!nodePhantomCanvases[nodeId]) {
        // Create foreignObject with canvas - ONE per node
        // Middle nodes: position above/below to avoid child overlap
        const xOffset = isMiddle ? NODE_WIDTH/2 - 30 : NODE_WIDTH/2 - 20;
        const yOffset = isMiddle ? -canvasHeight/2 - 20 : -canvasHeight/2;

        const fo = phantomG.append('foreignObject')
            .attr('class', `phantom-canvas-${nodeId}`)
            .attr('x', nodeX + xOffset)
            .attr('y', nodeY + yOffset)
            .attr('width', canvasWidth)
            .attr('height', canvasHeight)
            .style('pointer-events', 'none')
            .style('overflow', 'visible');

        const canvas = fo.append('xhtml:canvas')
            .attr('width', canvasWidth)
            .attr('height', canvasHeight)
            .attr('class', 'phantom-canvas-fade')
            .style('background', 'transparent')
            .style('opacity', 0);

        nodePhantomCanvases[nodeId] = canvas.node();
    }

    // Debounce redraws for performance - batch rapid updates
    if (nodePhantomDrawPending[nodeId]) {
        clearTimeout(nodePhantomDrawPending[nodeId]);
    }

    nodePhantomDrawPending[nodeId] = setTimeout(() => {
        requestAnimationFrame(() => drawPhantomCanvas(nodeId, canvasWidth, canvasHeight));
    }, phantomDrawThrottleMs);  // Throttle to prevent lag
}

function updatePhantomOrientationMode() {
    // Update phantom canvas visibility using CSS classes (fast, smooth transitions)
    if (!Object.keys(nodePhantomCanvases).length) return;  // Nothing to update

    const subtree = showOrientation ? getSubtreeIds(referenceNodeId) : null;

    Object.keys(nodePhantomCanvases).forEach(nodeId => {
        const canvas = nodePhantomCanvases[nodeId];
        if (!canvas) return;

        const nid = parseInt(nodeId);
        const isAncestor = showOrientation && subtree && !subtree.has(nid);

        // Use CSS class for smooth transitions
        d3.select(canvas).classed('orient-ancestor', isAncestor);
    });
}

function drawPhantomCanvas(nodeId, canvasWidth, canvasHeight) {
    const canvas = nodePhantomCanvases[nodeId];
    if (!canvas) return;

    const canvasD3 = d3.select(canvas);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    const centerY = canvasHeight / 2;
    const trajCount = nodePhantomData[nodeId]?.length || 0;

    // Reduce opacity as more trajectories accumulate (keep it readable)
    const accumFade = Math.max(0.4, 1.0 - trajCount * 0.06);

    // Draw all accumulated trajectories - MORE DRAMATIC
    nodePhantomData[nodeId].forEach((traj, trajIdx) => {
        const { words, topColors, isMiddle: trajIsMiddle } = traj;

        // Curves for dramatic effect - balanced for performance
        const numCurves = trajIsMiddle
            ? Math.min(3, 2 + Math.floor(words.length / 15))
            : Math.min(5, 3 + Math.floor(words.length / 12));

        for (let c = 0; c < numCurves; c++) {
            const curveColor = topColors[c % topColors.length];
            const { r, g, b } = curveColor;

            // MUCH MORE vertical spread - some curves go VERY far
            const verticalBias = trajIsMiddle ? 3.5 : 2.0;
            const isFarCurve = Math.random() > 0.7;  // 30% of curves go very far
            const baseAngle = (trajIdx * 0.5 + c * 0.3 - numCurves * 0.15) + (Math.random() - 0.5) * 2.5;

            // LONGER curves, especially "far" ones
            const length = trajIsMiddle
                ? (80 + Math.random() * 250 + words.length * 4)
                : (isFarCurve
                    ? (200 + Math.random() * 500 + words.length * 12)  // FAR curves
                    : (120 + Math.random() * 400 + words.length * 6));

            const startX = 15;
            const startY = centerY;
            const endX = startX + length;
            // FAR curves go much further vertically
            const verticalMultiplier = isFarCurve ? 200 : 120;
            const endY = centerY + baseAngle * verticalMultiplier * verticalBias;

            // More dramatic bezier curves - wilder control points
            const cp1x = startX + length * (0.2 + Math.random() * 0.15);
            const cp1y = centerY + (Math.random() - 0.5) * 280 * verticalBias;
            const cp2x = startX + length * (0.55 + Math.random() * 0.2);
            const cp2y = endY + (Math.random() - 0.5) * 180 * verticalBias;

            // Draw curve - subtle artistic
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, endX, endY);
            const curveOpacity = (0.02 + Math.random() * 0.04) * accumFade;
            ctx.strokeStyle = `rgba(${r},${g},${b},${curveOpacity})`;
            ctx.lineWidth = 0.5 + Math.random() * 2;
            if (Math.random() > 0.5) ctx.setLineDash([3 + Math.random()*4, 5 + Math.random()*5]);
            else ctx.setLineDash([]);
            ctx.stroke();
            ctx.setLineDash([]);

            // Draw words along curve - DRAMATIC FONT SIZE VARIATION
            // Skip some words for performance
            const wordStep = words.length > 20 ? 2 : 1;
            for (let w = 0; w < words.length; w += wordStep) {
                const word = words[w];
                const t = (w + 0.15 + Math.random()*0.25) / words.length;
                const mt = 1 - t;

                const wordColor = topColors[w % topColors.length];

                const wx = mt*mt*mt*startX + 3*mt*mt*t*cp1x + 3*mt*t*t*cp2x + t*t*t*endX;
                const wy = mt*mt*mt*startY + 3*mt*mt*t*cp1y + 3*mt*t*t*cp2y + t*t*t*endY + (Math.random()-0.5)*50*verticalBias;

                // DRAMATIC font size variation - some words HUGE, some tiny
                const sizeVariation = Math.random();
                let fontSize;
                if (sizeVariation > 0.92) {
                    // 8% chance of HUGE text
                    fontSize = 22 + Math.random() * 18;
                } else if (sizeVariation > 0.8) {
                    // 12% chance of large text
                    fontSize = 14 + Math.random() * 10;
                } else if (sizeVariation < 0.15) {
                    // 15% chance of tiny text
                    fontSize = 4 + Math.random() * 3;
                } else {
                    // Normal range with position-based scaling
                    fontSize = trajIsMiddle
                        ? (6 + Math.random()*6 + (1-t)*5)
                        : (8 + Math.random()*8 + (1-t)*6);
                }

                // Opacity also varies - bigger = more visible, but subtle overall
                const sizeOpacityBoost = Math.min(1, fontSize / 20);
                const wordOpacity = (0.03 + (1-t)*0.08 + Math.random()*0.05 + sizeOpacityBoost * 0.06) * accumFade;

                ctx.font = `italic ${Math.random() > 0.7 ? '500' : '300'} ${fontSize}px system-ui`;
                ctx.fillStyle = `rgba(${wordColor.r},${wordColor.g},${wordColor.b},${wordOpacity})`;
                ctx.save();
                ctx.translate(wx, wy);
                // More rotation for dramatic effect
                ctx.rotate((Math.random()-0.5)*0.6);
                ctx.fillText(word.slice(0,18), 0, 0);
                ctx.restore();
            }
        }

        // Particles, varied sizes - balanced for performance
        const numParticles = trajIsMiddle
            ? (4 + Math.floor(Math.random() * 6))
            : (8 + Math.floor(Math.random() * 10));
        for (let p = 0; p < numParticles; p++) {
            const particleColor = topColors[p % topColors.length];
            const px = 10 + Math.random() * (canvasWidth - 20);
            const py = centerY + (Math.random() - 0.5) * (canvasHeight * 0.95);
            const pr = 0.8 + Math.random() * (trajIsMiddle ? 3.5 : 4.5);

            ctx.beginPath();
            ctx.arc(px, py, pr, 0, Math.PI * 2);
            const particleOpacity = (0.03 + Math.random()*0.08) * accumFade;
            ctx.fillStyle = `rgba(${particleColor.r},${particleColor.g},${particleColor.b},${particleOpacity})`;
            ctx.fill();
        }
    });

    // FEATHER EDGES: Soft fade-out at edges - WIDER fade area
    ctx.globalCompositeOperation = 'destination-in';

    const gradient = ctx.createRadialGradient(
        canvasWidth * 0.12, centerY, 0,
        canvasWidth * 0.35, centerY, Math.max(canvasWidth, canvasHeight) * 0.8
    );
    gradient.addColorStop(0, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.4, 'rgba(255,255,255,1)');
    gradient.addColorStop(0.65, 'rgba(255,255,255,0.7)');
    gradient.addColorStop(0.85, 'rgba(255,255,255,0.3)');
    gradient.addColorStop(1, 'rgba(255,255,255,0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    ctx.globalCompositeOperation = 'source-over';

    // Smooth fade in
    canvasD3.style('opacity', 1);
}

// Batch node updates to prevent lag
const pendingNodeUpdates = {};
let nodeUpdateTimeout = null;
const NODE_UPDATE_BATCH_MS = 80;

function updateTreeNode(d){
    // Guard: tree must be initialized
    if (!nodeGroups || !treeG) {
        console.warn('⏳ updateTreeNode called before tree initialized, skipping node_id=', d.node_id);
        return;
    }

    // Queue update for batching
    pendingNodeUpdates[d.node_id] = d;

    // Schedule batched update
    if (!nodeUpdateTimeout) {
        nodeUpdateTimeout = setTimeout(() => {
            const updates = Object.values(pendingNodeUpdates);
            Object.keys(pendingNodeUpdates).forEach(k => delete pendingNodeUpdates[k]);
            nodeUpdateTimeout = null;

            // Process all queued updates
            updates.forEach(data => processNodeUpdate(data));

            // Single DOM update at the end
            updateNodeColors();
            updateTreeStats();
        }, NODE_UPDATE_BATCH_MS);
    }
}

function processNodeUpdate(d){
    const n = treeState?.nodes?.find(x => x.node_id === d.node_id);
    if(n){
        n.core=d.core;
        n.n_samples=d.n_samples;
        n.expected_relative_orientations=d.expected_relative_orientations;
        // Store trajectories and samples for exploration
        if (!n.trajectories) n.trajectories = [];
        if (!n.samples) n.samples = [];
        if (d.trajectory) {
            n.trajectories.push(d.trajectory);
            // Add dynamic phantom branch from this trajectory
            addDynamicPhantomBranch(d.node_id, d.trajectory, d.scores);
        }
        if (d.scores) n.samples.push(d.scores);
    } else {
        return;  // Skip update if node doesn't exist yet
    }
    treeState.total_samples=d.total_samples;
    treeState.total_api_calls=d.total_api_calls;
    const ng=nodeGroups.filter(x=>x.data.node_id===d.node_id);
    if (ng.empty()) {
        return;  // Skip if D3 node doesn't exist yet
    }

    const bw = (NODE_WIDTH/2) - 40;
    const subtree = showOrientation ? getSubtreeIds(referenceNodeId) : null;
    const inSubtree = !showOrientation || (subtree && subtree.has(d.node_id));
    ng.each(function(){
        const g=d3.select(this);
        const scores=d.core;
        if(!scores||!scores.length)return;
        const bh=Math.min(8, (BAR_AREA_HEIGHT - 20)/scores.length);
        const sy=-((scores.length*bh)/2);

        // Get or create bar-group
        let bg=g.select('.bar-group');
        if(bg.empty()){
            bg=g.append('g').attr('class','bar-group').attr('transform', 'translate(0, 5)');
            bg.append('line').attr('class','center-line')
                .attr('x1',0).attr('y1',sy-3)
                .attr('x2',0).attr('y2',sy+scores.length*bh+3)
                .attr('stroke','rgba(100,120,150,.4)');
            // Create initial bars at zero width
            scores.forEach((v,i)=>{
                bg.append('rect')
                    .attr('class','bar bar-'+i)
                    .attr('x',0)
                    .attr('y',sy+i*bh+1)
                    .attr('width',0)
                    .attr('height',bh-2)
                    .attr('fill',colors[i%colors.length])
                    .attr('rx',3);
            });
        }

        const expectedRelativeOrientation = showOrientation ? getNodeOrientation(d.node_id, referenceNodeId) : null;

        // Transition existing bars to new positions
        scores.forEach((v,i)=>{
            const relOrientation = expectedRelativeOrientation ? expectedRelativeOrientation[i] : null;
            const geom = CoreMath.computeBarGeometry(v, relOrientation, bw, showOrientation);
            const hi=highlightedQuestion===null||highlightedQuestion===i;
            const baseOpacity = inSubtree ? (hi ? 0.9 : 0.2) : 0.15;
            bg.select('.bar-'+i)
                .transition().duration(300)
                .attr('x',geom.x)
                .attr('width',geom.width)
                .attr('opacity',baseOpacity);
        });
    });
    // Note: updateNodeColors() and updateTreeStats() are called in batched mode
    // Refresh trajectories panel if showing this node
    if (currentTrajectoriesNodeId === d.node_id) {
        showTrajectories(d.node_id);
    }
}

function updateTreeStats(){
    if (!treeState?.nodes) return;  // Guard against early calls
    document.getElementById('statValue1').textContent=treeState.total_samples||0;
    document.getElementById('statValue2').textContent=treeState.nodes.length;
}

function setupTreeLegend(){
    const leg=document.getElementById('legend');
    leg.innerHTML='';
    treeState.questions.forEach((q,i)=>{
        const it=document.createElement('div');
        it.className='legend-item';
        it.innerHTML=`<div class="legend-swatch" style="background:${colors[i%colors.length]}"></div><span>${q}</span>`;
        it.onclick=()=>{
            highlightedQuestion=highlightedQuestion===i?null:i;
            document.querySelectorAll('.legend-item').forEach((el,j)=>el.classList.toggle('dimmed',highlightedQuestion!==null&&j!==i));
            drawTreeBars();
        };
        leg.appendChild(it);
    });
}

// ════════════════════════════════════════════════════════════════════════════════
// TREE PAGE - Mode toggle handlers
// ════════════════════════════════════════════════════════════════════════════════
document.getElementById('btnCore').onclick=()=>{
    showOrientation=false;
    document.getElementById('btnCore').classList.add('active');
    document.getElementById('btnOrient').classList.remove('active');
    requestAnimationFrame(() => {
        applyOrientationStyling();
        drawTreeBars();
        updateNodeColors();
        updatePhantomOrientationMode();
    });
};

document.getElementById('btnOrient').onclick=()=>{
    showOrientation=true;
    document.getElementById('btnOrient').classList.add('active');
    document.getElementById('btnCore').classList.remove('active');
    requestAnimationFrame(() => {
        applyOrientationStyling();
        drawTreeBars();
        updateNodeColors();
        updatePhantomOrientationMode();
    });
};
"""
