/**
 * TVB-O Knowledge Graph Visualization
 * D3.js force-directed graph for ontology + database items
 */

class KnowledgeGraphVisualization {
    constructor() {
        this.svg = null;
        this.simulation = null;
        this.nodes = [];
        this.links = [];
        this.width = 0;
        this.height = 0;
        this.zoom = null;
        this.g = null;
        this.selectedNode = null;
        this.limit = 100; // Default limit for performance

        // Color scheme for node types
        this.colors = {
            'Class': '#667eea',        // Ontology classes - blue
            'instance': '#48bb78',     // Database instances - green
            'dynamics': '#ed8936',     // Dynamics - orange
            'network': '#9f7aea',      // Networks - purple
            'integrator': '#38b2ac',   // Integrators - teal
            'experiment': '#f56565',   // Experiments - red
            'study': '#4299e1',        // Studies - light blue
            'coupling': '#ed64a6',     // Couplings - pink
            'ontology': '#667eea',     // Generic ontology - blue
        };
    }

    async init() {
        const container = document.getElementById('graphContainer');
        const svgEl = document.getElementById('graphSvg');
        if (!container || !svgEl) return;

        this.width = container.clientWidth;
        this.height = container.clientHeight || 600;

        this.svg = d3.select('#graphSvg')
            .attr('width', this.width)
            .attr('height', this.height);

        // Clear previous content
        this.svg.selectAll('*').remove();

        // Add zoom behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on('zoom', (event) => {
                this.g.attr('transform', event.transform);
            });

        this.svg.call(this.zoom);

        // Create main group for zooming
        this.g = this.svg.append('g');

        // Add arrow marker for directed edges
        this.svg.append('defs').append('marker')
            .attr('id', 'arrowhead')
            .attr('viewBox', '-0 -5 10 10')
            .attr('refX', 20)
            .attr('refY', 0)
            .attr('orient', 'auto')
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .append('path')
            .attr('d', 'M 0,-5 L 10,0 L 0,5')
            .attr('fill', '#999');

        // Setup controls
        this.setupControls();

        // Load and render graph
        await this.loadGraph();
    }

    setupControls() {
        document.getElementById('graphZoomIn')?.addEventListener('click', () => {
            this.svg.transition().duration(300).call(this.zoom.scaleBy, 1.3);
        });

        document.getElementById('graphZoomOut')?.addEventListener('click', () => {
            this.svg.transition().duration(300).call(this.zoom.scaleBy, 0.7);
        });

        document.getElementById('graphReset')?.addEventListener('click', () => {
            this.svg.transition().duration(500).call(
                this.zoom.transform,
                d3.zoomIdentity.translate(this.width / 2, this.height / 2).scale(0.5)
            );
        });
    }

    async loadGraph() {
        try {
            this.showLoading();

            // Use limit parameter for better performance
            const url = `/tvbo/api/kg/graph?limit=${this.limit}`;
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data = await response.json();

            this.nodes = data.nodes || [];
            this.links = data.links || [];

            if (this.nodes.length === 0) {
                this.showError('No graph data available');
                return;
            }

            this.render();
            this.renderLegend(data.stats || {});
        } catch (error) {
            console.error('Failed to load graph:', error);
            this.showError(`Failed to load graph: ${error.message}`);
        }
    }

    showLoading() {
        const container = document.getElementById('graphContainer');
        if (container) {
            // Clear any existing error messages but keep SVG structure
            const existing = container.querySelector('.graph-loading');
            if (!existing) {
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'graph-loading';
                loadingDiv.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 10; text-align: center;';
                loadingDiv.innerHTML = `
                    <div style="font-size: 2rem;">⏳</div>
                    <p>Loading knowledge graph...</p>
                `;
                container.style.position = 'relative';
                container.appendChild(loadingDiv);
            }
        }
    }

    hideLoading() {
        const loading = document.querySelector('.graph-loading');
        if (loading) loading.remove();
    }

    showError(message) {
        const container = document.getElementById('graphContainer');
        if (container) {
            container.innerHTML = `
                <div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #e53e3e;">
                    <div style="text-align: center;">
                        <div style="font-size: 3rem;">⚠️</div>
                        <p>${message}</p>
                        <button onclick="window.kgBrowser?.graphViz?.init()" style="margin-top: 10px; padding: 8px 16px; cursor: pointer;">Retry</button>
                    </div>
                </div>
            `;
        }
    }

    render() {
        this.hideLoading();

        // Create link elements
        const link = this.g.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(this.links)
            .join('line')
            .attr('stroke', d => d.type === 'instance_of' ? '#48bb78' : '#999')
            .attr('stroke-opacity', 0.6)
            .attr('stroke-width', d => d.type === 'instance_of' ? 2 : 1)
            .attr('marker-end', 'url(#arrowhead)');

        // Create node elements
        const node = this.g.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(this.nodes)
            .join('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', (event, d) => this.dragStarted(event, d))
                .on('drag', (event, d) => this.dragged(event, d))
                .on('end', (event, d) => this.dragEnded(event, d)));

        // Add circles to nodes
        node.append('circle')
            .attr('r', d => this.getNodeRadius(d))
            .attr('fill', d => this.getNodeColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2);

        // Add labels
        node.append('text')
            .text(d => this.truncateLabel(d.label || d.name || ''))
            .attr('x', 0)
            .attr('y', d => this.getNodeRadius(d) + 12)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('fill', '#4a5568');

        // Add tooltip on hover
        node.on('mouseover', (event, d) => this.showTooltip(event, d))
            .on('mouseout', () => this.hideTooltip())
            .on('click', (event, d) => this.handleNodeClick(event, d));

        // Create force simulation
        this.simulation = d3.forceSimulation(this.nodes)
            .force('link', d3.forceLink(this.links)
                .id(d => d.storid || d.id)
                .distance(80))
            .force('charge', d3.forceManyBody().strength(-200))
            .force('center', d3.forceCenter(this.width / 2, this.height / 2))
            .force('collision', d3.forceCollide().radius(d => this.getNodeRadius(d) + 5));

        // Update positions on tick
        this.simulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            node.attr('transform', d => `translate(${d.x},${d.y})`);
        });
    }

    getNodeRadius(node) {
        if (node.type === 'instance') return 6;
        if (node.db_type) return 8;
        return 10; // Ontology classes
    }

    getNodeColor(node) {
        if (node.db_type) return this.colors[node.db_type] || this.colors.instance;
        if (node.type === 'instance') return this.colors.instance;
        return this.colors[node.type] || this.colors.Class;
    }

    truncateLabel(text, maxLen = 15) {
        if (text.length <= maxLen) return text;
        return text.slice(0, maxLen - 1) + '…';
    }

    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }

    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }

    showTooltip(event, node) {
        let tooltip = document.getElementById('graphTooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'graphTooltip';
            tooltip.className = 'graph-tooltip';
            document.body.appendChild(tooltip);
        }

        const typeLabel = node.db_type || node.type || 'Class';
        tooltip.innerHTML = `
            <div class="tooltip-title">${node.label || node.name || 'Unknown'}</div>
            <div class="tooltip-type">${typeLabel}</div>
            ${node.iri ? `<div class="tooltip-iri">${node.iri}</div>` : ''}
        `;
        tooltip.style.left = (event.pageX + 10) + 'px';
        tooltip.style.top = (event.pageY - 10) + 'px';
        tooltip.classList.add('visible');
    }

    hideTooltip() {
        const tooltip = document.getElementById('graphTooltip');
        if (tooltip) tooltip.classList.remove('visible');
    }

    handleNodeClick(event, node) {
        event.stopPropagation();

        // Highlight connected nodes
        const connectedIds = new Set();
        this.links.forEach(l => {
            const sourceId = l.source.storid || l.source.id || l.source;
            const targetId = l.target.storid || l.target.id || l.target;
            const nodeId = node.storid || node.id;

            if (sourceId === nodeId) connectedIds.add(targetId);
            if (targetId === nodeId) connectedIds.add(sourceId);
        });

        this.g.selectAll('.node circle')
            .attr('opacity', d => {
                const dId = d.storid || d.id;
                const nodeId = node.storid || node.id;
                return dId === nodeId || connectedIds.has(dId) ? 1 : 0.2;
            });

        this.g.selectAll('.links line')
            .attr('opacity', l => {
                const sourceId = l.source.storid || l.source.id || l.source;
                const targetId = l.target.storid || l.target.id || l.target;
                const nodeId = node.storid || node.id;
                return sourceId === nodeId || targetId === nodeId ? 1 : 0.1;
            });

        // Open modal with details
        if (window.kgBrowser && node.db_type) {
            const dbItem = window.kgBrowser.originalData.find(item =>
                item.id === node.id?.replace(`db_${node.db_type}_`, '') ||
                `db_${item.type}_${item.id}` === node.id
            );
            if (dbItem) window.kgBrowser.openModal(dbItem);
        }
    }

    resetHighlight() {
        this.g.selectAll('.node circle').attr('opacity', 1);
        this.g.selectAll('.links line').attr('opacity', 0.6);
    }

    renderLegend(stats) {
        const legend = document.getElementById('graphLegend');
        if (!legend) return;

        const types = [
            { label: 'Ontology Class', color: this.colors.Class },
            { label: 'Dynamics', color: this.colors.dynamics },
            { label: 'Network', color: this.colors.network },
            { label: 'Integrator', color: this.colors.integrator },
            { label: 'Experiment', color: this.colors.experiment },
            { label: 'Study', color: this.colors.study },
            { label: 'Coupling', color: this.colors.coupling },
        ];

        legend.innerHTML = `
            <div class="legend-title">Legend</div>
            ${types.map(t => `
                <div class="legend-item">
                    <span class="legend-color" style="background:${t.color}"></span>
                    <span class="legend-label">${t.label}</span>
                </div>
            `).join('')}
            <div class="legend-stats">
                <div>Classes: ${stats.ontology_classes}</div>
                <div>Instances: ${stats.database_items}</div>
                <div>Links: ${stats.total_links}</div>
            </div>
        `;
    }

    destroy() {
        if (this.simulation) {
            this.simulation.stop();
        }
        this.svg?.selectAll('*').remove();
    }
}

// Export for use in kg_browser.js
window.KnowledgeGraphVisualization = KnowledgeGraphVisualization;
