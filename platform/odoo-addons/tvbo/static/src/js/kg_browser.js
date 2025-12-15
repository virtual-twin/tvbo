/**
 * TVB-O Knowledge Graph Browser
 * API-powered faceted search interface
 */

// Render Markdown while protecting LaTeX math
function renderMarkdownWithMath(md) {
    if (!md || typeof md !== 'string') return '';

    const codePlaceholders = [];
    let codeIdx = 0;
    const protectCode = (text) => {
        text = text.replace(/```[\s\S]*?```/g, (m) => {
            const key = `@@CODE_BLOCK_${codeIdx++}@@`;
            codePlaceholders.push({ key, val: m });
            return key;
        });
        text = text.replace(/`[^`]*`/g, (m) => {
            const key = `@@CODE_INLINE_${codeIdx++}@@`;
            codePlaceholders.push({ key, val: m });
            return key;
        });
        return text;
    };

    const restoreCode = (html) => {
        codePlaceholders.forEach(({ key, val }) => {
            html = html.split(key).join(val);
        });
        return html;
    };

    const mathPlaceholders = [];
    let mathIdx = 0;
    const protectMath = (text) => {
        text = text.replace(/\$\$[\s\S]*?\$\$/g, (m) => {
            const key = `@@MATH_BLOCK_${mathIdx++}@@`;
            mathPlaceholders.push({ key, val: m });
            return key;
        });
        let out = '';
        for (let i = 0; i < text.length; i++) {
            if (text[i] === '$') {
                if (text[i + 1] === '$') {
                    out += '$$';
                    i += 1;
                    continue;
                }
                let j = i + 1;
                let found = -1;
                while (j < text.length) {
                    if (text[j] === '$' && text[j - 1] !== '\\') { found = j; break; }
                    j++;
                }
                if (found !== -1) {
                    const segment = text.slice(i, found + 1);
                    const key = `@@MATH_INLINE_${mathIdx++}@@`;
                    mathPlaceholders.push({ key, val: segment });
                    out += key;
                    i = found;
                } else {
                    out += text[i];
                }
            } else {
                out += text[i];
            }
        }
        return out;
    };

    let safe = protectCode(md);
    safe = protectMath(safe);

    let html = (window.marked && window.marked.parse)
        ? window.marked.parse(safe)
        : safe
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br/>');

    mathPlaceholders.forEach(({ key, val }) => {
        html = html.split(key).join(val);
    });
    html = restoreCode(html);

    return html;
}

class KnowledgeGraphBrowser {
    constructor() {
        console.log('🏗️ KnowledgeGraphBrowser constructor called');
        this.originalData = [];
        this.schema = null;
        this.currentFilters = {};
        this.currentQuery = '';
        this.collapsedFacets = new Set();
        this.lastResultIndices = [];
        this.currentSort = 'relevance';
        this.currentModalItem = null;
        this.searchIndex = new Map();
        this.facetIndex = {};
        this.currentView = 'list'; // 'list' or 'graph'
        this.graphViz = null;

        this.init();
    }

    async init() {
        try {
            console.log('📡 Fetching data from API...');
            await this.loadData();
            console.log('📚 Building search index...');
            this.searchIndex = this.buildSearchIndex();
            console.log('✅ Search index built:', this.searchIndex.size, 'tokens');
            console.log('🏷️ Building facet index...');
            this.facetIndex = this.buildFacetIndex();
            console.log('✅ Facet index built');
            console.log('🎧 Setting up event listeners...');
            this.setupEventListeners();

            // Check for search query in URL (from header search)
            const urlParams = new URLSearchParams(window.location.search);
            const queryFromUrl = urlParams.get('q');
            if (queryFromUrl) {
                console.log('🔗 Found search query in URL:', queryFromUrl);
                const searchBox = document.getElementById('searchBox');
                if (searchBox) {
                    searchBox.value = queryFromUrl;
                }
                this.currentQuery = queryFromUrl;
            }

            console.log('🔍 Performing initial search...');
            this.search();
            console.log('✅ Browser initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing browser:', error);
            this.showError('Failed to load data. Please try refreshing the page.');
        }
    }

    async loadData() {
        // Fetch schema
        const schemaResponse = await fetch('/tvbo/api/kg/schema');
        if (!schemaResponse.ok) {
            throw new Error('Failed to fetch schema');
        }
        this.schema = await schemaResponse.json();
        console.log('📊 Schema loaded:', this.schema);

        // Fetch data
        const dataResponse = await fetch('/tvbo/api/kg/data');
        if (!dataResponse.ok) {
            throw new Error('Failed to fetch data');
        }
        this.originalData = await dataResponse.json();
        console.log('📦 Data loaded:', this.originalData.length, 'items');

        if (this.originalData.length === 0) {
            console.warn('⚠️ No data returned from API');
        }
    }

    showError(message) {
        const resultsGrid = document.getElementById('resultsGrid');
        if (resultsGrid) {
            resultsGrid.innerHTML = `
                <div class="no-results">
                    <div style="font-size: 4rem; margin-bottom: 20px;">⚠️</div>
                    <h3>Error</h3>
                    <p>${message}</p>
                </div>
            `;
        }
        const perfInfo = document.getElementById('performanceInfo');
        if (perfInfo) {
            perfInfo.textContent = 'Error loading data';
        }
    }

    buildSearchIndex() {
        console.log('Building search index for', this.originalData.length, 'items');
        const index = new Map();
        this.originalData.forEach((item, idx) => {
            const searchText = this.schema.searchableFields
                .map(field => {
                    const value = item[field];
                    if (value === undefined || value === null) {
                        return '';
                    }
                    if (Array.isArray(value)) {
                        return value.join(' ');
                    }
                    return String(value);
                })
                .join(' ')
                .toLowerCase();
            const tokens = searchText.split(/\s+/).filter(token => token.length > 0);
            tokens.forEach(token => {
                if (!index.has(token)) {
                    index.set(token, new Set());
                }
                index.get(token).add(idx);
            });
        });
        console.log('Search index created with', index.size, 'unique tokens');
        return index;
    }

    buildFacetIndex() {
        console.log('Building facet index for fields:', this.schema.facets.map(f => f.field));
        const index = {};
        this.schema.facets.forEach(facet => {
            index[facet.field] = new Map();
            console.log(`Processing facet: ${facet.field} (type: ${facet.type})`);
            this.originalData.forEach((item, idx) => {
                const value = item[facet.field];
                if (value === undefined || value === null || value === '') {
                    return;
                }
                if (facet.type === 'array') {
                    if (!Array.isArray(value)) {
                        const arrayValue = [value];
                        arrayValue.forEach(val => {
                            if (val !== undefined && val !== null && val !== '') {
                                const key = String(val);
                                if (!index[facet.field].has(key)) {
                                    index[facet.field].set(key, new Set());
                                }
                                index[facet.field].get(key).add(idx);
                            }
                        });
                        return;
                    }
                    value.forEach(val => {
                        if (val !== undefined && val !== null && val !== '') {
                            const key = String(val);
                            if (!index[facet.field].has(key)) {
                                index[facet.field].set(key, new Set());
                            }
                            index[facet.field].get(key).add(idx);
                        }
                    });
                } else {
                    const key = String(value);
                    if (!index[facet.field].has(key)) {
                        index[facet.field].set(key, new Set());
                    }
                    index[facet.field].get(key).add(idx);
                }
            });
            console.log(`Facet ${facet.field} indexed with`, index[facet.field].size, 'unique values');
        });
        return index;
    }

    setupEventListeners() {
        let searchTimeout;
        const searchBox = document.getElementById('searchBox');
        const searchForm = document.getElementById('globalSearchForm');

        // Prevent form submission on KG browser page - use live search instead
        if (searchForm) {
            searchForm.addEventListener('submit', (e) => {
                e.preventDefault();
                if (searchBox) {
                    this.currentQuery = searchBox.value;
                    this.search();
                }
            });
        }

        if (searchBox) {
            searchBox.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.currentQuery = e.target.value;
                    this.search();
                }, 150);
            });
        }

        document.addEventListener('click', (e) => {
            if (e.target.closest('.facet-item')) {
                this.handleFacetClick(e.target.closest('.facet-item'));
            } else if (e.target.closest('.facet-header')) {
                this.handleFacetToggle(e.target.closest('.facet-header'));
            } else if (e.target.closest('.result-card')) {
                const card = e.target.closest('.result-card');
                const idxRaw = card?.dataset?.idx;
                const idx = parseInt(idxRaw);
                if (!isNaN(idx)) {
                    const item = this.originalData[idx];
                    this.openModal(item);
                }
            }
        });

        const clearBtn = document.getElementById('clearFilters');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearAllFilters();
            });
        }

        const sortSel = document.getElementById('sortSelect');
        if (sortSel) {
            sortSel.addEventListener('change', () => {
                this.currentSort = sortSel.value || 'relevance';
                const items = this.lastResultIndices.map(idx => this.originalData[idx]);
                this.renderResults(items);
            });
        }

        // Modal handlers
        const modal = document.getElementById('detailsModal');
        const closeBtn = document.getElementById('modalClose');
        const backdrop = modal ? modal.querySelector('.modal-backdrop') : null;
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.closeModal());
        }
        if (backdrop) {
            backdrop.addEventListener('click', () => this.closeModal());
        }
        const downloadBtn = document.getElementById('modalDownload');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadModalItem());
        }

        // View toggle handlers
        const listViewBtn = document.getElementById('listViewBtn');
        const graphViewBtn = document.getElementById('graphViewBtn');

        if (listViewBtn) {
            listViewBtn.addEventListener('click', () => this.switchView('list'));
        }
        if (graphViewBtn) {
            graphViewBtn.addEventListener('click', () => this.switchView('graph'));
        }
    }

    async switchView(view) {
        this.currentView = view;

        const resultsGrid = document.getElementById('resultsGrid');
        const graphContainer = document.getElementById('graphContainer');
        const listViewBtn = document.getElementById('listViewBtn');
        const graphViewBtn = document.getElementById('graphViewBtn');

        if (view === 'list') {
            resultsGrid?.classList.remove('hidden');
            graphContainer?.classList.add('hidden');
            listViewBtn?.classList.add('active');
            graphViewBtn?.classList.remove('active');

            // Cleanup graph
            if (this.graphViz) {
                this.graphViz.destroy();
                this.graphViz = null;
            }
        } else {
            resultsGrid?.classList.add('hidden');
            graphContainer?.classList.remove('hidden');
            listViewBtn?.classList.remove('active');
            graphViewBtn?.classList.add('active');

            // Initialize graph visualization
            if (window.KnowledgeGraphVisualization) {
                this.graphViz = new window.KnowledgeGraphVisualization();
                await this.graphViz.init();
            }
        }
    }

    handleFacetClick(facetItem) {
        const filterKey = facetItem.dataset.filter;
        const value = facetItem.dataset.value;

        if (!this.currentFilters[filterKey]) {
            this.currentFilters[filterKey] = [];
        }

        const index = this.currentFilters[filterKey].indexOf(value);
        if (index > -1) {
            this.currentFilters[filterKey].splice(index, 1);
            if (this.currentFilters[filterKey].length === 0) {
                delete this.currentFilters[filterKey];
            }
        } else {
            this.currentFilters[filterKey].push(value);
        }

        this.search();
    }

    handleFacetToggle(header) {
        const group = header.parentElement;
        const field = group.dataset.field;
        if (this.collapsedFacets.has(field)) {
            this.collapsedFacets.delete(field);
            group.classList.remove('collapsed');
        } else {
            this.collapsedFacets.add(field);
            group.classList.add('collapsed');
        }
    }

    clearAllFilters() {
        this.currentFilters = {};
        this.currentQuery = '';
        const searchBox = document.getElementById('searchBox');
        if (searchBox) {
            searchBox.value = '';
        }
        this.search();
    }

    search() {
        const startTime = performance.now();
        let results = new Set();

        // Text search
        if (this.currentQuery && this.currentQuery.length > 0) {
            const query = this.currentQuery.toLowerCase();
            const tokens = query.split(/\s+/).filter(token => token.length > 0);

            if (tokens.length === 0) {
                results = new Set(this.originalData.map((_, idx) => idx));
            } else {
                tokens.forEach((token, tokenIdx) => {
                    const matching = new Set();
                    this.searchIndex.forEach((indices, indexToken) => {
                        if (indexToken.includes(token)) {
                            indices.forEach(idx => matching.add(idx));
                        }
                    });

                    if (tokenIdx === 0) {
                        results = matching;
                    } else {
                        results = new Set([...results].filter(idx => matching.has(idx)));
                    }
                });
            }
        } else {
            results = new Set(this.originalData.map((_, idx) => idx));
        }

        // Apply filters
        Object.entries(this.currentFilters).forEach(([field, values]) => {
            if (values && values.length > 0) {
                const matchingIndices = new Set();
                values.forEach(value => {
                    const indices = this.facetIndex[field]?.get(value);
                    if (indices) {
                        indices.forEach(idx => matchingIndices.add(idx));
                    }
                });
                results = new Set([...results].filter(idx => matchingIndices.has(idx)));
            }
        });

        const resultItems = [...results].map(idx => this.originalData[idx]);
        this.lastResultIndices = [...results];

        const endTime = performance.now();
        const searchTime = (endTime - startTime).toFixed(2);

        this.renderFacets(results);
        this.renderResults(resultItems);

        const perfInfo = document.getElementById('performanceInfo');
        if (perfInfo) {
            perfInfo.textContent = `Found ${resultItems.length} results in ${searchTime}ms`;
        }
    }

    renderFacets(currentResults) {
        const facetsSidebar = document.getElementById('facetsSidebar');
        if (!facetsSidebar) return;
        facetsSidebar.innerHTML = '';

        this.schema.facets.forEach(facet => {
            const facetValues = new Map();

            currentResults.forEach(idx => {
                const item = this.originalData[idx];
                const value = item[facet.field];

                if (facet.type === 'array' && Array.isArray(value)) {
                    value.forEach(val => {
                        if (val) {
                            const key = String(val);
                            facetValues.set(key, (facetValues.get(key) || 0) + 1);
                        }
                    });
                } else if (value !== undefined && value !== null && value !== '') {
                    const key = String(value);
                    facetValues.set(key, (facetValues.get(key) || 0) + 1);
                }
            });

            if (facetValues.size === 0) return;

            const facetGroup = document.createElement('div');
            facetGroup.className = 'facet-group';
            facetGroup.dataset.field = facet.field;
            if (this.collapsedFacets.has(facet.field)) {
                facetGroup.classList.add('collapsed');
            }

            const header = document.createElement('div');
            header.className = 'facet-header';
            header.innerHTML = `<span class="facet-title">${facet.label}</span>`;
            facetGroup.appendChild(header);

            const content = document.createElement('div');
            content.className = 'facet-content';

            const sortedValues = [...facetValues.entries()].sort((a, b) => b[1] - a[1]);

            sortedValues.forEach(([value, count]) => {
                const facetItem = document.createElement('div');
                facetItem.className = 'facet-item';
                facetItem.dataset.filter = facet.field;
                facetItem.dataset.value = value;

                const isActive = this.currentFilters[facet.field] &&
                                this.currentFilters[facet.field].includes(value);
                if (isActive) {
                    facetItem.classList.add('active');
                }

                facetItem.innerHTML = `
                    <input type="checkbox" class="facet-checkbox" ${isActive ? 'checked' : ''}>
                    <span class="facet-label">${value}</span>
                    <span class="facet-count">${count}</span>
                `;

                content.appendChild(facetItem);
            });

            facetGroup.appendChild(content);
            facetsSidebar.appendChild(facetGroup);
        });
    }

    renderResults(items) {
        const resultsGrid = document.getElementById('resultsGrid');
        const resultsCount = document.getElementById('resultsCount');

        if (resultsCount) {
            resultsCount.textContent = `${items.length} result${items.length !== 1 ? 's' : ''}`;
        }

        if (!resultsGrid) return;

        if (items.length === 0) {
            resultsGrid.innerHTML = `
                <div class="no-results">
                    <div style="font-size: 4rem; margin-bottom: 20px;">🔍</div>
                    <h3>No items found</h3>
                    <p>Try adjusting your filters or search terms</p>
                </div>
            `;
            return;
        }

        let sortedItems = items;
        if (this.currentSort === 'az' || this.currentSort === 'za') {
            sortedItems = [...items]
                .map((it, i) => ({ it, idx: this.lastResultIndices[i] }))
                .sort((a, b) => {
                    const ta = String(this.getItemTitle(a.it) || '').toLowerCase();
                    const tb = String(this.getItemTitle(b.it) || '').toLowerCase();
                    const cmp = ta.localeCompare(tb);
                    return this.currentSort === 'az' ? cmp : -cmp;
                })
                .map(x => x.it);
        }

        resultsGrid.innerHTML = sortedItems.map((item, i) => {
            const originalIdx = this.lastResultIndices[i];
            const title = this.getItemTitle(item);
            const desc = this.truncateText(this.getItemDescription(item), 150);
            const type = (item.type || '').toLowerCase();
            const isOntology = type === 'ontology';
            const ontologyType = item.ontology_type || '';

            // Determine icon and styling
            let typeIcon, typeClass, typeLabel;
            if (isOntology) {
                typeIcon = 'schema';
                typeClass = 'type-ontology';
                typeLabel = ontologyType || 'Ontology';
            } else {
                typeIcon = type === 'model' ? 'functions'
                             : type === 'dynamics' ? 'functions'
                             : type === 'study' ? 'article'
                             : type === 'network' ? 'device_hub'
                             : type === 'coupling' ? 'compare_arrows'
                             : type === 'integrator' ? 'speed'
                             : type === 'experiment' ? 'science'
                             : 'label';
                typeClass = type ? `type-${type}` : '';
                typeLabel = item.type ? item.type : '';
            }

            const typeBadge = typeLabel ? `<span class="badge ${typeClass}"><span class="material-icons">${typeIcon}</span>${typeLabel}</span>` : '';
            const studyMeta = (item.type === 'study') ? this.getStudyMeta(item) : '';

            // Show symbol for ontology items
            const symbolDisplay = isOntology && item.symbol && item.symbol !== title
                ? `<span class="ontology-symbol">${item.symbol}</span>` : '';

            const header = `
                <div class="field-display" style="margin-bottom:6px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                    <span class="field-value" style="font-weight:700; color:#2d3748; font-size:1.05rem;">${this.escapeHtml(title)}</span>
                    ${symbolDisplay}
                    ${typeBadge}
                </div>`;
            const body = desc ? `
                <div class="card-body">
                    <div class="card-desc"><span class="field-value" style="color:#4a5568;">${this.escapeHtml(desc)}</span></div>
                </div>` : '';

            // Show tags if available
            const tags = item.tags && item.tags.length > 0
                ? `<div class="card-tags">${item.tags.map(t => `<span class="tag-pill">${this.escapeHtml(t)}</span>`).join('')}</div>`
                : '';

            return `
                <div class="result-card" data-idx="${originalIdx}">
                    ${header}
                    ${studyMeta}
                    ${body}
                    ${tags}
                </div>
            `;
        }).join('');
    }

    getStudyMeta(item) {
        const parts = [];
        if (item.year) parts.push(String(item.year));
        if (item.journal) parts.push(item.journal);
        if (item.doi) {
            const doiUrl = item.doi.startsWith('http') ? item.doi : `https://doi.org/${item.doi}`;
            parts.push(`<a href="${doiUrl}" target="_blank" onclick="event.stopPropagation()">doi</a>`);
        }
        if (parts.length === 0) return '';
        return `<div class="field-display"><span class="field-value" style="color:#718096; font-size: 0.9rem;">${parts.join(' · ')}</span></div>`;
    }

    getItemTitle(item) {
        return item.title || item.name || item.label || item.id || 'Untitled';
    }

    truncateText(text, maxLen = 200) {
        if (!text) return '';
        const clean = String(text).replace(/\s+/g, ' ').trim();
        if (clean.length <= maxLen) return clean;
        const cutoff = clean.lastIndexOf(' ', maxLen);
        if (cutoff > Math.floor(maxLen * 0.6)) {
            return clean.slice(0, cutoff).trimEnd() + '…';
        }
        return clean.slice(0, maxLen).trimEnd() + '…';
    }

    getItemDescription(item) {
        return item.description || item.desc || item.summary || item.abstract || '';
    }

    async openModal(item) {
        const modal = document.getElementById('detailsModal');
        if (!modal) return;
        this.currentModalItem = item;
        const titleEl = document.getElementById('modalTitle');
        const contentEl = document.getElementById('modalContent');

        if (titleEl) titleEl.textContent = this.getItemTitle(item);

        if (contentEl) {
            // Fetch detailed data if available
            let detailData = item;
            const isOntology = item.type === 'ontology';

            try {
                if (isOntology && item.storid) {
                    const resp = await fetch(`/tvbo/api/kg/ontology/node/${item.storid}`);
                    if (resp.ok) {
                        detailData = await resp.json();
                    }
                } else if (item.type === 'model' && item.id) {
                    const resp = await fetch(`/tvbo/api/kg/model/${item.id}`);
                    if (resp.ok) {
                        detailData = await resp.json();
                    }
                } else if (item.type === 'dynamics' && item.id) {
                    const resp = await fetch(`/tvbo/api/kg/dynamics/${item.id}`);
                    if (resp.ok) {
                        detailData = await resp.json();
                    }
                } else if (item.type === 'network' && item.id) {
                    const resp = await fetch(`/tvbo/api/kg/network/${item.id}`);
                    if (resp.ok) {
                        detailData = await resp.json();
                    }
                }
            } catch (e) {
                console.warn('Could not fetch detailed data:', e);
            }

            let html = '';

            // Ontology-specific header with symbol and IRI
            if (isOntology) {
                if (detailData.symbol && detailData.symbol !== detailData.label) {
                    html += `<div class="ontology-header">
                        <span class="ontology-symbol-large">${detailData.symbol}</span>
                    </div>`;
                }
                if (detailData.iri) {
                    html += `<div class="ontology-iri">
                        <a href="${detailData.iri}" target="_blank" rel="noopener">${detailData.iri}</a>
                    </div>`;
                }
            }

            // Description / Definition
            const descText = detailData.definition || detailData.description;
            if (descText) {
                html += `<div class="modal-section">
                    <div class="detail-value">${renderMarkdownWithMath(descText)}</div>
                </div>`;
            }

            // Type badge and metadata
            html += '<div class="modal-details">';

            // Basic fields
            const basicFields = isOntology
                ? ['ontology_type', 'type', 'iri']
                : ['type', 'system_type', 'source', 'iri', 'year', 'journal', 'doi'];
            basicFields.forEach(key => {
                const value = detailData[key];
                if (value !== undefined && value !== null && value !== '' && key !== 'iri') {
                    html += `<div class="detail-row">
                        <div class="detail-label">${this.formatLabel(key)}:</div>
                        <div class="detail-value">${this.escapeHtml(String(value))}</div>
                    </div>`;
                }
            });

            // Ontology hierarchy (is_a)
            if (isOntology && detailData.is_a && detailData.is_a.length > 0) {
                html += `<div class="detail-row">
                    <div class="detail-label">Parent Classes:</div>
                    <div class="detail-value">
                        ${detailData.is_a.map(p =>
                            `<span class="badge type-ontology">${this.escapeHtml(p)}</span>`
                        ).join(' ')}
                    </div>
                </div>`;
            }

            // Ontology requirements
            if (isOntology && detailData.requires && detailData.requires.length > 0) {
                html += `<div class="detail-row">
                    <div class="detail-label">Requires:</div>
                    <div class="detail-value">
                        ${detailData.requires.map(r =>
                            `<span class="badge">${this.escapeHtml(String(r))}</span>`
                        ).join(' ')}
                    </div>
                </div>`;
            }

            // Parameters
            if (detailData.parameters && detailData.parameters.length > 0) {
                html += `<div class="detail-row">
                    <div class="detail-label">Parameters:</div>
                    <div class="detail-value">
                        ${detailData.parameters.map(p =>
                            `<span class="badge">${this.escapeHtml(p.name || p.label || '')}</span>`
                        ).join(' ')}
                    </div>
                </div>`;
            }

            // State variables
            if (detailData.state_variables && detailData.state_variables.length > 0) {
                html += `<div class="detail-row">
                    <div class="detail-label">State Variables:</div>
                    <div class="detail-value">
                        ${detailData.state_variables.map(sv => {
                            let svHtml = `<span class="badge">${this.escapeHtml(sv.label || '')}</span>`;
                            if (sv.equation && sv.equation.definition) {
                                svHtml += `<div class="eq-display">${sv.equation.definition}</div>`;
                            }
                            return svHtml;
                        }).join(' ')}
                    </div>
                </div>`;
            }

            // Tags
            if (detailData.tags && detailData.tags.length > 0) {
                html += `<div class="detail-row">
                    <div class="detail-label">Tags:</div>
                    <div class="detail-value">
                        ${detailData.tags.map(t =>
                            `<span class="badge">${this.escapeHtml(t)}</span>`
                        ).join(' ')}
                    </div>
                </div>`;
            }

            // References
            if (detailData.references) {
                html += `<div class="detail-row">
                    <div class="detail-label">References:</div>
                    <div class="detail-value">${renderMarkdownWithMath(detailData.references)}</div>
                </div>`;
            }

            html += '</div>';
            contentEl.innerHTML = html;

            // Typeset math
            if (window.MathJax && window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise([contentEl]).catch(() => {});
            }
        }

        modal.classList.remove('hidden');
        modal.classList.add('show');
        modal.style.display = 'block';
    }

    closeModal() {
        const modal = document.getElementById('detailsModal');
        if (!modal) return;
        modal.classList.add('hidden');
        modal.classList.remove('show');
        modal.style.display = 'none';
        this.currentModalItem = null;
    }

    downloadModalItem() {
        if (!this.currentModalItem) return;
        const yaml = this.objectToYAML(this.currentModalItem);
        const filename = (this.currentModalItem.title || this.currentModalItem.name || 'item').replace(/\s+/g, '_') + '.yaml';
        const blob = new Blob([yaml], { type: 'text/yaml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    objectToYAML(obj, indent = 0) {
        const pad = '  '.repeat(indent);
        let yaml = '';

        Object.entries(obj).forEach(([key, value]) => {
            if (value === undefined || value === null) return;

            yaml += `${pad}${key}:`;

            if (Array.isArray(value)) {
                yaml += '\n';
                value.forEach(item => {
                    if (typeof item === 'object') {
                        yaml += `${pad}- ${this.objectToYAML(item, indent + 1).trim()}\n`;
                    } else {
                        yaml += `${pad}- ${item}\n`;
                    }
                });
            } else if (typeof value === 'object') {
                yaml += '\n' + this.objectToYAML(value, indent + 1);
            } else {
                yaml += ` ${value}\n`;
            }
        });

        return yaml;
    }

    formatLabel(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎬 DOM ready, initializing Knowledge Graph Browser...');
    window.kgBrowser = new KnowledgeGraphBrowser();
});
