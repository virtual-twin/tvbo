// TVB LinkML Faceted Browser JS
// Full implementation with modals and event handlers

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

class OptimizedFacetedSearch {
    constructor(data, schema) {
        console.log('🏗️ OptimizedFacetedSearch constructor called');
        console.log('Data sample:', data.slice(0, 2));
        console.log('Schema:', schema);
        this.originalData = data;
        this.schema = schema;
        this.currentFilters = {};
        this.currentQuery = '';
        this.collapsedFacets = new Set();
        this.sliderTimeout = null;
        this.lastResultIndices = [];
        this.currentSort = 'relevance';
        this.currentModalItem = null;

        try {
            console.log('📚 Building search index...');
            this.searchIndex = this.buildSearchIndex();
            console.log('✅ Search index built:', this.searchIndex.size, 'tokens');
            console.log('🏷️ Building facet index...');
            this.facetIndex = this.buildFacetIndex();
            console.log('✅ Facet index built');
            console.log('🎧 Setting up event listeners...');
            this.setupEventListeners();
            console.log('🔍 Performing initial search...');
            this.search();
            console.log('✅ Constructor complete');
        } catch (error) {
            console.error('❌ Error in constructor:', error);
            throw error;
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
                        console.warn(`Missing field ${field} in item:`, item);
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
                if (value === undefined || value === null) {
                    console.warn(`Missing facet field ${facet.field} in item ${idx}:`, item);
                    return;
                }
                if (facet.type === 'array') {
                    if (!Array.isArray(value)) {
                        console.warn(`Expected array for ${facet.field} in item ${idx}, got:`, typeof value, value);
                        const arrayValue = [value];
                        arrayValue.forEach(val => {
                            if (val !== undefined && val !== null) {
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
                        if (val !== undefined && val !== null) {
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
        document.getElementById('searchBox').addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.currentQuery = e.target.value;
                this.search();
            }, 150);
        });

        document.addEventListener('click', (e) => {
            console.log('🖱️ Click event:', { target: e.target, classList: e.target.classList });

            if (e.target.closest('.facet-item')) {
                console.log('📌 Facet item clicked');
                this.handleFacetClick(e.target.closest('.facet-item'));
            } else if (e.target.closest('.facet-header')) {
                console.log('📂 Facet header clicked');
                this.handleFacetToggle(e.target.closest('.facet-header'));
            } else if (e.target.closest('.result-card')) {
                const card = e.target.closest('.result-card');
                const idxRaw = card?.dataset?.idx;
                const idx = parseInt(idxRaw);
                console.log('🟦 Card click detected', {
                    target: e.target,
                    card,
                    idxRaw,
                    idx,
                    cardClasses: card?.className,
                    hasOriginalData: !!this.originalData,
                    dataLength: this.originalData?.length
                });
                if (isNaN(idx)) {
                    console.warn('⚠️ Card click: invalid idx', { idxRaw, card });
                } else {
                    const item = this.originalData[idx];
                    console.log('📄 Opening details modal for', item?.title || item?.name || item?.label || idx);
                    this.openModal(item);
                }
            }
        });

        document.getElementById('clearFilters').addEventListener('click', () => {
            this.clearAllFilters();
        });

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
        document.getElementById('searchBox').value = '';
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
                    const indices = this.facetIndex[field].get(value);
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
        facetsSidebar.innerHTML = '';

        this.schema.facets.forEach(facet => {
            const facetValues = new Map();

            currentResults.forEach(idx => {
                const item = this.originalData[idx];
                const value = item[facet.field];

                if (facet.type === 'array' && Array.isArray(value)) {
                    value.forEach(val => {
                        const key = String(val);
                        facetValues.set(key, (facetValues.get(key) || 0) + 1);
                    });
                } else if (value !== undefined && value !== null) {
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

        resultsCount.textContent = `${items.length} result${items.length !== 1 ? 's' : ''}`;

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
            const desc = this.getItemDescription(item);
            const type = (item.type || '').toLowerCase();
            const typeIcon = type === 'schema' ? 'schema'
                             : type === 'model' ? 'functions'
                             : type === 'study' ? 'article'
                             : type === 'atlas' ? 'map'
                             : type === 'network' ? 'device_hub'
                             : type === 'coupling' ? 'compare_arrows'
                             : 'label';
            const typeClass = type ? `type-${type}` : '';
            const typeLabel = item.type ? item.type : '';
            const typeBadge = typeLabel ? `<span class="badge ${typeClass}"><span class="material-icons">${typeIcon}</span>${typeLabel}</span>` : '';
            const studyMeta = (item.type === 'study') ? this.getStudyMeta(item) : '';
            const thumb = item.thumbnail ? `<img class="thumb" src="${item.thumbnail}" alt="thumbnail" onerror="this.style.display='none'">` : '';
            const header = `
                <div class="field-display" style="margin-bottom:6px; display:flex; align-items:center; gap:8px;">
                    <span class="field-value" style="font-weight:700; color:#2d3748; font-size:1.05rem;">${title}</span>
                    ${typeBadge}
                </div>`;
            const body = desc ? `
                <div class="card-body">
                    ${thumb}
                    <div class="card-desc"><span class="field-value" style="color:#4a5568;">${desc}</span></div>
                </div>` : (thumb ? `
                <div class="card-body">${thumb}</div>` : '');
            return `
                <div class="result-card" data-idx="${originalIdx}">
                    ${header}
                    ${studyMeta}
                    ${body}
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
            parts.push(`<a href="${doiUrl}" target="_blank">doi</a>`);
        }
        if (parts.length === 0) return '';
        return `<div class="field-display"><span class="field-value" style="color:#718096; font-size: 0.9rem;">${parts.join(' · ')}</span></div>`;
    }

    getItemTitle(item) {
        return item.title || item.name || item.label || item.id || item.file || 'Untitled';
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



    openModal(item) {
        const modal = document.getElementById('detailsModal');
        if (!modal) return;
        this.currentModalItem = item;
        const titleEl = document.getElementById('modalTitle');
        const contentEl = document.getElementById('modalContent');
        const dlBtn = document.getElementById('modalDownload');
        if (titleEl) titleEl.textContent = this.getItemTitle(item);
        if (dlBtn) {
            dlBtn.onclick = () => {
                const yaml = this.objectToYAML(item);
                const filename = (item.title || item.name || 'item').replace(/\s+/g, '_') + '.yaml';
                const blob = new Blob([yaml], { type: 'text/yaml' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            };
        }
        if (contentEl) {
            let html = '';
            // Thumbnail at top if available
            if (item.thumbnail) {
                html += `
                    <div class="field-display" style="margin:6px 0 10px; display:flex; justify-content:center;">
                        <img src="${item.thumbnail}" alt="thumbnail" style="max-width:100%; border-radius:6px; box-shadow:0 1px 3px rgba(0,0,0,0.15)" onerror="this.style.display='none'"/>
                    </div>`;
            }
            // Model report
            if (item.type === 'model' && item.report_md) {
                try {
                    html += `<div class="model-report">${renderMarkdownWithMath(item.report_md)}</div>`;
                } catch (e) {
                    console.error('Error rendering model report:', e);
                }
            }
            // Display all fields
            // html += '<div class="modal-details">';
            // Object.entries(item).forEach(([key, value]) => {
            //     if (value !== undefined && value !== null && key !== 'id' && key !== 'report_md' && key !== 'thumbnail' && key !== 'full_model') {
            //         html += `<div class="detail-row">`;
            //         html += `<div class="detail-label">${this.escapeHtml(key)}:</div>`;
            //         if (Array.isArray(value)) {
            //             html += `<div class="detail-value">${value.map(v =>
            //                 `<span class="badge">${this.escapeHtml(String(v))}</span>`
            //             ).join(' ')}</div>`;
            //         } else if (typeof value === 'object') {
            //             html += `<div class="detail-value"><pre>${this.escapeHtml(JSON.stringify(value, null, 2))}</pre></div>`;
            //         } else {
            //             html += `<div class="detail-value">${this.escapeHtml(String(value))}</div>`;
            //         }
            //         html += `</div>`;
            //     }
            // });
            // html += '</div>';
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

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when data is ready
function initializeSearch() {
    console.log('🔍 Initializing search...');
    const data = window.searchData;
    const schema = window.searchSchema;

    console.log('📊 Data check:', typeof data);

    if (!data || !schema) {
        console.error('❌ Missing data or schema');
        return;
    }

    if (!Array.isArray(data) || data.length === 0) {
        console.error('❌ Invalid data format');
        return;
    }

    console.log('🏷️ Schema searchableFields:', schema.searchableFields);
    console.log('📊 Schema facets:', schema.facets);

    // Check for missing fields
    const missingFields = schema.searchableFields.filter(field =>
        !data[0].hasOwnProperty(field)
    );
    if (missingFields.length > 0) {
        console.warn('⚠️ Missing searchable fields:', missingFields);
    }

    const missingFacets = schema.facets.filter(facet =>
        !data[0].hasOwnProperty(facet.field)
    );
    if (missingFacets.length > 0) {
        console.warn('⚠️ Missing facet fields:', missingFacets.map(f => f.field));
    }

    console.log('🚀 Creating search instance...');
    window.searchInstance = new OptimizedFacetedSearch(data, schema);
    console.log('✅ Search initialized successfully');
}

// Wait for data to be ready
console.log('🔧 Setting up event listeners...');
if (window.searchData && window.searchSchema) {
    console.log('📦 Data already loaded, initializing immediately');
    initializeSearch();
} else {
    window.addEventListener('searchDataReady', function() {
        console.log('📡 searchDataReady event received');
        initializeSearch();
    });
}

console.log('🎬 DOM ready, starting browser initialization...');
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎬 DOM ready, starting browser initialization...');

    if (window.searchData && window.searchSchema) {
        console.log('📦 Data already loaded, initializing immediately');
        initializeSearch();
    }
});
