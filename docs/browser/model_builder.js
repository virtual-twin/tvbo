// Model Builder UI for TVB-O Browser
// Creates an interactive modal to assemble a brain network model spec
// and export it as YAML/JSON. No external dependencies.

(function () {
  // Determine asset base path from this script's location (e.g., /docs/_site/browser/)
  const SCRIPT_DIR = (() => {
    const current = document.currentScript || Array.from(document.getElementsByTagName('script')).find(el => (el.src || '').includes('model_builder.js'));
    if (!current || !current.src) return '';
    try {
      const url = new URL(current.src, window.location.href);
      const path = url.pathname;
      return path.replace(/[^/]+$/,'');
    } catch {
      // Fallback: try to infer from location
      const loc = window.location.pathname || '';
      return loc.endsWith('browser.html') ? loc.replace(/browser\.html$/,'') + 'browser/' : '';
    }
  })();
  const ASSET = (name) => (SCRIPT_DIR ? (SCRIPT_DIR + name) : name);
  const STATE = {
    dataLoaded: false,
    data: [],
  lastFullModelSpec: null,
  previewEnabled: false,
  };

  const DEBUG = true;
  const log = (...args) => { if (DEBUG && console && console.log) console.log('[ModelBuilder]', ...args); };

  function whenDataReady(cb) {
    if (window.searchData && Array.isArray(window.searchData)) {
      STATE.dataLoaded = true;
      STATE.data = window.searchData;
  log('searchData ready immediately:', { count: STATE.data.length });
      try {
        const ms = STATE.data.filter(x => (x.type||'').toLowerCase()==='model');
        const withFull = ms.filter(m => !!m.full_spec_url).length;
        log('Models summary:', { total: ms.length, withFullSpec: withFull, withoutFullSpec: ms.length - withFull });
      } catch(_) {}
      cb(window.searchData);
      return;
    }
    window.addEventListener('searchDataReady', () => {
      STATE.dataLoaded = true;
      STATE.data = window.searchData || [];
  log('searchData event:', { count: STATE.data.length });
      try {
        const ms = STATE.data.filter(x => (x.type||'').toLowerCase()==='model');
        const withFull = ms.filter(m => !!m.full_spec_url).length;
        log('Models summary:', { total: ms.length, withFullSpec: withFull, withoutFullSpec: ms.length - withFull });
      } catch(_) {}
      cb(STATE.data);
    });
  }

  function createButton() {
    const header = document.querySelector('.header');
    if (!header) return;

    const btn = document.createElement('button');
    btn.textContent = 'Open Model Builder';
    btn.className = 'modal-btn';
    btn.style.marginLeft = 'auto';
    btn.style.marginTop = '8px';

    const wrap = document.createElement('div');
    wrap.style.display = 'flex';
    wrap.style.justifyContent = 'flex-end';
    wrap.appendChild(btn);
    header.appendChild(wrap);

    btn.addEventListener('click', () => openBuilderModal());
  }

  async function ensureModalRoot() {
    // Ensure CSS is present
    const hasBuilderCss = Array.from(document.querySelectorAll('link[rel="stylesheet"]')).some(l => (l.href || '').includes('builder.css'));
    if (!hasBuilderCss) {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = ASSET('builder.css');
      link.setAttribute('data-builder-css', '1');
      document.head.appendChild(link);
    }

    // Try to load external HTML/template once
    let modal = document.getElementById('builderModal');
    let needTpl = !document.getElementById('builderContentTpl');

    try {
      const res = await fetch(ASSET('builder.html'), { cache: 'no-cache' });
      if (res.ok) {
        const html = await res.text();
        const tmp = document.createElement('div');
        tmp.innerHTML = html;
        const extModal = tmp.querySelector('#builderModal');
        const tpl = tmp.querySelector('#builderContentTpl');
        if (tpl && !document.getElementById('builderContentTpl')) {
          document.body.appendChild(tpl); // keep template in DOM for cloning
        }
        if (!modal && extModal) {
          document.body.appendChild(extModal);
          modal = extModal;
          const closeBtn = modal.querySelector('#builderClose');
          const backdrop = modal.querySelector('.modal-backdrop');
          closeBtn && closeBtn.addEventListener('click', () => closeBuilderModal());
          backdrop && backdrop.addEventListener('click', () => closeBuilderModal());
          return modal;
        }
        // If template was loaded and modal already existed, return modal now
        if (modal) return modal;
      }
    } catch(_) {
      // ignore and fall back
    }

    // Fallback: minimal modal chrome if external HTML not available
    modal = document.createElement('div');
    modal.id = 'builderModal';
    modal.className = 'modal hidden';
    modal.innerHTML = `
      <div class="modal-backdrop"></div>
      <div class="modal-dialog" style="max-width: 980px;">
        <div class="modal-header">
          <div class="modal-title">Brain Network Model Builder</div>
          <div class="modal-actions">
            <button class="modal-btn" id="builderCopyPython">Copy Python</button>
            <button class="modal-btn" id="builderDownloadYaml">Download YAML</button>
            <button class="modal-close" id="builderClose" aria-label="Close">×</button>
          </div>
        </div>
        <div class="modal-content" id="builderContent"></div>
      </div>`;
    document.body.appendChild(modal);
    modal.querySelector('#builderClose').addEventListener('click', () => closeBuilderModal());
    modal.querySelector('.modal-backdrop').addEventListener('click', () => closeBuilderModal());
    return modal;
  }

  async function openBuilderModal() {
    // Ensure MathJax is available for previews (best-effort)
    if (!window.MathJax && !document.getElementById('MathJax-script')) {
      const s = document.createElement('script');
      s.id = 'MathJax-script';
      s.async = true;
      s.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';
      document.head.appendChild(s);
    }
    const modal = await ensureModalRoot();
    const content = modal.querySelector('#builderContent');
    renderBuilder(content);
  log('Opened builder modal');
    // Ensure visibility regardless of Bootstrap's .modal default display:none
    modal.classList.remove('hidden');
    modal.classList.add('show');
    modal.style.display = 'block';
  }

  function closeBuilderModal() {
    const modal = document.getElementById('builderModal');
    if (modal) {
      modal.classList.add('hidden');
      modal.classList.remove('show');
      modal.style.display = 'none';
    }
  }

  function renderBuilder(root) {
    const data = STATE.data || [];

    const models = data.filter(x => (x.type || '').toLowerCase() === 'model');
    const networks = data.filter(x => (x.type || '').toLowerCase() === 'network');
    // Prefer dedicated coupling items (loaded from data/CouplingFunctions), else fallback by name contains
    let couplings = data.filter(x => (x.type || '').toLowerCase() === 'coupling');
    if (couplings.length === 0) {
      couplings = data.filter(x => (String(x.title || x.name || '').toLowerCase().includes('coupling')));
    }
    // Clear root and clone external template if available
    root.innerHTML = '';
    const tpl = document.getElementById('builderContentTpl');
    if (tpl && tpl.content) {
      root.appendChild(tpl.content.cloneNode(true));
    }

    const section = root; // use root as the container for queries below

  // Dynamic rows helpers (new ordered sections)
  const mpRows = section.querySelector('#modelParamsRows');
  const dpRows = section.querySelector('#derivedParamsRows');
  const fnRows = section.querySelector('#functionsRows');
  const seRows = section.querySelector('#stateEqRows');
  const dvRows = section.querySelector('#derivedVarsRows');
  const otRows = section.querySelector('#outputTransformsRows');
  const cpRows = section.querySelector('#couplingParamsRows');

  // Use STATE.lastFullModelSpec to persist full model JSON for spec assembly

  function rowEquation(name = '', expr = '') {
      const row = document.createElement('div');
      row.className = 'builder-row';
      row.style.alignItems = 'center';
      row.style.gridTemplateColumns = '1fr 1fr auto';
      row.innerHTML = `
        <input class="builder-input eq-name" placeholder="name (e.g., dV/dt)" value="${escapeAttr(name)}" />
        <div class="eq-expr-wrap" style="display:flex; flex-direction:column; gap:4px; align-items:flex-start; width:100%;">
          <input class="builder-input eq-expr" placeholder="expression (e.g., -a*V + I)" value="${escapeAttr(expr)}" />
        </div>
        <button class="modal-btn eq-del" title="Remove">✕</button>
        <div class="eq-preview" style="grid-column: 1 / 3; display:block; width:100%; font-size: 0.6em; line-height: 1.2; color:#1f2937; white-space:normal; word-break:break-word; overflow:visible; min-height: 0;"></div>`;
      const update = () => updateEquationPreview(row);
      row.querySelector('.eq-del').addEventListener('click', () => row.remove());
      row.querySelector('.eq-name').addEventListener('input', update);
      row.querySelector('.eq-expr').addEventListener('input', update);
      // initial render
  // Respect current toggle state (stack with full-width preview below)
  const wrap = row.querySelector('.eq-expr-wrap');
  const input = row.querySelector('.eq-expr');
  const prev = row.querySelector('.eq-preview');
  wrap.style.flexDirection = 'column';
  wrap.style.alignItems = 'flex-start';
  wrap.style.gap = '4px';
  input.style.flex = '1 1 auto';
  prev.style.display = STATE.previewEnabled ? 'block' : 'none';
  prev.style.width = '100%';
      update();
      return row;
    }

    // Specific equation row helpers per section (aliases to rowEquation for now)
    const rowDerivedParam = (name = '', expr = '') => rowEquation(name, expr);
    const rowFunction = (name = '', expr = '') => rowEquation(name, expr);
    const rowStateEq = (name = '', expr = '') => rowEquation(name, expr);
    const rowDerivedVar = (name = '', expr = '') => rowEquation(name, expr);
    const rowOutputTransform = (name = '', expr = '') => rowEquation(name, expr);

    function rowParam(name = '', value = '', unit = '', description = '') {
      const frag = document.createDocumentFragment();
      const row = document.createElement('div');
      row.className = 'builder-row';
      row.style.gridTemplateColumns = '1fr 1fr 1fr auto';
      row.innerHTML = `
        <input class="builder-input p-name" placeholder="name" value="${escapeAttr(name)}" />
        <input class="builder-input p-value" placeholder="value" value="${escapeAttr(value)}" />
        <div class="p-unit-wrap" style="display:flex; align-items:center; gap:6px;">
          <input class="builder-input p-unit" placeholder="unit (optional)" value="${escapeAttr(unit)}" />
          <button class="info-btn p-info" title="Show description" style="display:${description ? 'inline-block' : 'none'};">i</button>
        </div>
        <button class="modal-btn p-del" title="Remove">✕</button>`;
      const desc = document.createElement('div');
      desc.className = 'param-desc';
      desc.textContent = description || '';
      const infoBtn = row.querySelector('.p-info');
      if (infoBtn) {
        infoBtn.addEventListener('click', (e) => {
          e.preventDefault();
          desc.style.display = (desc.style.display === 'block') ? 'none' : 'block';
        });
      }
      row.querySelector('.p-del').addEventListener('click', () => { desc.remove(); row.remove(); });
      frag.appendChild(row);
      frag.appendChild(desc);
      return frag;
    }

    // Populate selects from data
    const modelSelectInit = section.querySelector('#builderModel');
    if (modelSelectInit) {
      const opts = ['<option value="">— select local dynamics —</option>']
        .concat(models.map(m => `<option value="${escapeHtml(m.key || m.id || m.name || m.title)}">${escapeHtml(m.title || m.name || m.id || String(m.key || 'model'))}</option>`));
      modelSelectInit.innerHTML = opts.join('');
    }
    const couplingSelectInit = section.querySelector('#builderCoupling');
    if (couplingSelectInit) {
      const opts = ['<option value="">— select coupling —</option>']
        .concat(couplings.map(c => `<option value="${escapeHtml(c.key || c.id || c.name || c.title)}">${escapeHtml(c.title || c.name || c.id || String(c.key || 'coupling'))}</option>`));
      couplingSelectInit.innerHTML = opts.join('');
    }

    // Seed with initial rows for each section
    if (mpRows) mpRows.appendChild(rowParam());
    if (dpRows) dpRows.appendChild(rowDerivedParam());
    if (fnRows) fnRows.appendChild(rowFunction());
    if (seRows) seRows.appendChild(rowStateEq());
    if (dvRows) dvRows.appendChild(rowDerivedVar());
    if (otRows) otRows.appendChild(rowOutputTransform());
    if (cpRows) cpRows.appendChild(rowParam());

    // Wire add buttons
    const addModelParamBtn = section.querySelector('#addModelParam');
    addModelParamBtn && mpRows && addModelParamBtn.addEventListener('click', () => mpRows.appendChild(rowParam()));
    const addDerivedParamBtn = section.querySelector('#addDerivedParam');
    addDerivedParamBtn && dpRows && addDerivedParamBtn.addEventListener('click', () => dpRows.appendChild(rowDerivedParam()));
    const addFunctionBtn = section.querySelector('#addFunction');
    addFunctionBtn && fnRows && addFunctionBtn.addEventListener('click', () => fnRows.appendChild(rowFunction()));
    const addStateEqBtn = section.querySelector('#addStateEquation');
    addStateEqBtn && seRows && addStateEqBtn.addEventListener('click', () => seRows.appendChild(rowStateEq()));
    const addDerivedVarBtn = section.querySelector('#addDerivedVariable');
    addDerivedVarBtn && dvRows && addDerivedVarBtn.addEventListener('click', () => dvRows.appendChild(rowDerivedVar()));
    const addOutputTransformBtn = section.querySelector('#addOutputTransform');
    addOutputTransformBtn && otRows && addOutputTransformBtn.addEventListener('click', () => otRows.appendChild(rowOutputTransform()));
    // Toggle LaTeX preview (applies to all equation-like rows)
    const previewToggle = section.querySelector('#toggleEqPreview');
    if (previewToggle) {
      previewToggle.checked = STATE.previewEnabled;
      previewToggle.addEventListener('change', () => {
        STATE.previewEnabled = !!previewToggle.checked;
        // update all rows across sections
        section.querySelectorAll('#derivedParamsRows .builder-row, #functionsRows .builder-row, #stateEqRows .builder-row, #derivedVarsRows .builder-row, #outputTransformsRows .builder-row').forEach(row => {
          const wrap = row.querySelector('.eq-expr-wrap');
          const input = row.querySelector('.eq-expr');
          const prev = row.querySelector('.eq-preview');
          wrap.style.flexDirection = 'column';
          wrap.style.alignItems = 'flex-start';
          wrap.style.gap = '4px';
          input.style.flex = '1 1 auto';
          prev.style.display = STATE.previewEnabled ? 'block' : 'none';
          prev.style.width = '100%';
          updateEquationPreview(row);
        });
      });
    }
    const addCouplingParamBtn = section.querySelector('#addCouplingParam');
    addCouplingParamBtn && cpRows && addCouplingParamBtn.addEventListener('click', () => cpRows.appendChild(rowParam()));

    // Coupling preview toggle and input handlers
    const couplingPreviewToggle = section.querySelector('#toggleCouplingEqPreview');
    if (couplingPreviewToggle) {
      couplingPreviewToggle.addEventListener('change', () => updateCouplingPreviews(section));
    }
    ['#couplingPreLhs', '#couplingPreRhs', '#couplingPostLhs', '#couplingPostRhs'].forEach(sel => {
      const el = section.querySelector(sel);
      if (el) el.addEventListener('input', () => { updateCouplingPreviews(section); try { drawCouplingPlot(section); } catch(_) {} });
    });
    const gxMin = section.querySelector('#gxMin');
    const gxMax = section.querySelector('#gxMax');
    [gxMin, gxMax].forEach(el => el && el.addEventListener('input', () => { try { drawCouplingPlot(section); } catch(_) {} }));

    // Auto-fill on model/coupling change
    const modelSelect = section.querySelector('#builderModel');
  const couplingSelect = section.querySelector('#builderCoupling');
  const couplingNameInput = section.querySelector('#builderCouplingName');
  const couplingSparseEl = section.querySelector('#couplingSparse');
  const couplingDelayedEl = section.querySelector('#couplingDelayed');
  const networkSelect = section.querySelector('#builderNetwork');
  const tractSelect = section.querySelector('#builderTract');
  const atlasSelect = section.querySelector('#builderAtlas');
  const netModeExisting = section.querySelector('#netModeExisting');
  const netModeToy = section.querySelector('#netModeToy');
  const netExistingWrap = section.querySelector('#netExistingWrap');
  const netToyWrap = section.querySelector('#netToyWrap');
  const netInfo = section.querySelector('#netInfo');
  const netThumb = section.querySelector('#netThumb');
  const netLabel = section.querySelector('#netLabel');
  const netRegions = section.querySelector('#netRegions');
  const netMean = section.querySelector('#netMean');
  const netMinMax = section.querySelector('#netMinMax');
  const netHist = section.querySelector('#netHist');
  const netDetailsLink = section.querySelector('#netDetailsLink');

    modelSelect && modelSelect.addEventListener('change', async () => {
      const key = modelSelect.value;
      const item = models.find(x => (x.key || x.id || x.name || x.title) == key);
      log('Model changed:', { key, itemHasFull: !!item?.full_spec_url, itemKeys: item ? Object.keys(item) : [] });
  // clear rows
  if (mpRows) mpRows.innerHTML = '';
  if (dpRows) dpRows.innerHTML = '';
  if (fnRows) fnRows.innerHTML = '';
  if (seRows) seRows.innerHTML = '';
  if (dvRows) dvRows.innerHTML = '';
  if (otRows) otRows.innerHTML = '';
      if (!item) {
        mpRows && mpRows.appendChild(rowParam());
        dpRows && dpRows.appendChild(rowDerivedParam());
        fnRows && fnRows.appendChild(rowFunction());
        seRows && seRows.appendChild(rowStateEq());
        dvRows && dvRows.appendChild(rowDerivedVar());
        otRows && otRows.appendChild(rowOutputTransform());
        return;
      }
      let full = item;
      if (item.full_model && typeof item.full_model === 'object') {
        STATE.lastFullModelSpec = item.full_model;
        full = Object.assign({}, item.full_model, item);
        log('Using embedded full_model for', key);
      } else if (item.full_spec_url) {
        try {
          const res = await fetch(item.full_spec_url);
          log('Fetching full_spec_url', item.full_spec_url, 'status:', res.status);
          if (res.ok) {
            const json = await res.json();
            // Merge basic visible fields over the full object for consistency
            full = Object.assign({}, json, item);
    STATE.lastFullModelSpec = json;
            log('Fetched full model spec keys:', Object.keys(json));
          }
        } catch (_) { /* ignore fetch errors */ }
      }
      // Fallback: try guessed models_json/<stem>.json if not loaded
      if (!STATE.lastFullModelSpec) {
        const candidates = guessModelJsonUrls(item);
        for (const url of candidates) {
          try {
            const res = await fetch(url);
            log('Trying fallback JSON', url, 'status:', res.status);
            if (res.ok) {
              const json = await res.json();
              STATE.lastFullModelSpec = json;
              full = Object.assign({}, json, item);
              break;
            }
          } catch(_) { /* ignore */ }
        }
      }
  if (!STATE.lastFullModelSpec) STATE.lastFullModelSpec = full;
      // Prefill sections from full model
      const params = extractParameters(full?.parameters || (full.model && full.model.parameters));
      if (mpRows) {
        if (params.length) params.forEach(p => mpRows.appendChild(rowParam(p.name || '', valueToStr(p.value), p.unit || '', p.description || '')));
        else mpRows.appendChild(rowParam());
      }

      const derivedParams = extractDerivedParameters(full);
      if (dpRows) {
        if (derivedParams.length) derivedParams.forEach(dp => dpRows.appendChild(rowDerivedParam(dp.name || '', dp?.equation?.rhs || '')));
        else dpRows.appendChild(rowDerivedParam());
      }

      // Functions
      const fns = extractFunctions(full);
      if (fnRows) {
        if (fns.length) fns.forEach(fn => fnRows.appendChild(rowFunction(fn.name || '', fn.rhs || fn.definition || '')));
        else fnRows.appendChild(rowFunction());
      }

      // State equations
      const ses = extractStateEquations(full);
      if (seRows) {
        if (ses.length) ses.forEach(e => seRows.appendChild(rowStateEq(e.lhs || e.label || '', e.rhs || e.definition || '')));
        else seRows.appendChild(rowStateEq());
      }

      // Derived variables
      const dvs = extractDerivedVariables(full);
      if (dvRows) {
        if (dvs.length) dvs.forEach(v => dvRows.appendChild(rowDerivedVar(v.name || '', v?.equation?.rhs || '')));
        else dvRows.appendChild(rowDerivedVar());
      }

      // Output transforms
      const outs = extractOutputTransforms(full);
      if (otRows) {
        if (outs.length) outs.forEach(o => otRows.appendChild(rowOutputTransform(o.name || '', o?.equation?.rhs || o.output || '')));
        else otRows.appendChild(rowOutputTransform());
      }
    });

    function refreshNetworkModeUI() {
      const isExisting = netModeExisting?.checked;
      const isToy = netModeToy?.checked;
  if (netExistingWrap) netExistingWrap.style.display = isExisting ? 'block' : 'none';
      if (netToyWrap) netToyWrap.style.display = isToy ? 'grid' : 'none';
      const toyType = section.querySelector('#toyType');
      const toyProbWrap = section.querySelector('#toyProbWrap');
      if (toyType && toyProbWrap) toyProbWrap.style.display = toyType.value === 'erdos' ? 'block' : 'none';
    }
  [netModeExisting, netModeToy].forEach(el => el && el.addEventListener('change', refreshNetworkModeUI));
    const toyTypeSel = section.querySelector('#toyType');
    if (toyTypeSel) toyTypeSel.addEventListener('change', refreshNetworkModeUI);
    refreshNetworkModeUI();
    // Populate tractogram and atlas options
    function populateNetworkFacets() {
      if (!Array.isArray(networks)) return;
      const tracts = new Map();
      const atlases = new Map();
      networks.forEach(n => {
        const t = (n.tractogram || n.desc || '').toString().trim();
        const a = (n.atlas || n?.parcellation?.atlas?.name || '').toString().trim();
        if (t) tracts.set(t, true);
        if (a) atlases.set(a, true);
      });
      const tractOpts = ['<option value="">— select tractogram —</option>'].concat(Array.from(tracts.keys()).sort().map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`));
      const atlasOpts = ['<option value="">— select atlas —</option>'].concat(Array.from(atlases.keys()).sort().map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`));
      if (tractSelect) tractSelect.innerHTML = tractOpts.join('');
      if (atlasSelect) atlasSelect.innerHTML = atlasOpts.join('');
    }
    populateNetworkFacets();

    function selectNetworkFromFacets() {
      const t = tractSelect?.value || '';
      const a = atlasSelect?.value || '';
      if (!t && !a) return;
      // Find first matching item by tractogram and/or atlas name
      const item = networks.find(n => {
        const tn = (n.tractogram || n.desc || '').toString().trim();
        const an = (n.atlas || n?.parcellation?.atlas?.name || '').toString().trim();
        return (!t || tn === t) && (!a || an === a);
      });
      // Trigger fill using existing logic via coupling/network change paths
      if (item) {
        // Store a synthetic key on the fly
        const key = item.key || item.id || item.name || item.title || `${a}:${t}`;
        // Reflect into hidden networkSelect if present to reuse mapping later
        if (networkSelect) networkSelect.value = key;
        updateNetworkInfo(item);
      }
    }
    tractSelect && tractSelect.addEventListener('change', selectNetworkFromFacets);
    atlasSelect && atlasSelect.addEventListener('change', selectNetworkFromFacets);

    function drawMiniHistogram(canvas, hist) {
      if (!canvas || !hist || !Array.isArray(hist.counts) || hist.counts.length === 0) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      const w = canvas.width = canvas.clientWidth || 180;
      const h = canvas.height = canvas.clientHeight || 60;
      ctx.clearRect(0,0,w,h);
      const counts = hist.counts;
      const maxC = Math.max(1, ...counts);
      const barW = Math.max(1, Math.floor((w - 8) / counts.length));
      const x0 = 4;
      ctx.fillStyle = '#93c5fd';
      counts.forEach((c,i) => {
        const bh = Math.round((c / maxC) * (h - 6));
        const x = x0 + i * barW;
        ctx.fillRect(x, h - bh - 2, Math.max(1, barW - 2), bh);
      });
      ctx.strokeStyle = '#94a3b8';
      ctx.strokeRect(0.5,0.5,w-1,h-1);
    }

    function updateNetworkInfo(item) {
      if (!netInfo) return;
      try {
        netInfo.style.display = 'flex';
        netThumb.src = (item.thumbnail || '').replace(/\\/g,'/');
        netThumb.style.display = item.thumbnail ? 'block' : 'none';
        netLabel.textContent = item.label || item.name || 'Network';
        netRegions.textContent = String(item.n_regions || '—');
        const mean = item.weights_mean;
        const min = item.weights_min;
        const max = item.weights_max;
        netMean.textContent = (mean !== undefined) ? Number(mean).toFixed(3) : '—';
        netMinMax.textContent = (min !== undefined && max !== undefined) ? `${Number(min).toFixed(3)} … ${Number(max).toFixed(3)}` : '—';
        drawMiniHistogram(netHist, item.weights_histogram);
        if (netDetailsLink) {
          netDetailsLink.onclick = () => showNetworkDetailsModal(item);
        }
      } catch(_) {}
    }

    function showNetworkDetailsModal(item) {
      const modal = document.createElement('div');
      modal.className = 'modal-sm';
      modal.innerHTML = `
        <div class="backdrop"></div>
        <div class="dialog">
          <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
            <div style="font-weight:700; color:#111827;">${escapeHtml(item.label || item.name || 'Network details')}</div>
            <button class="modal-btn" id="closeSm">Close</button>
          </div>
          <div class="mini-kv" style="margin-bottom:8px;">
            <div>Atlas:</div><div>${escapeHtml(item.atlas || '')}</div>
            <div>Tractogram:</div><div>${escapeHtml(item.desc || item.tractogram || '')}</div>
            <div>Regions:</div><div>${escapeHtml(String(item.n_regions || '—'))}</div>
            <div>Space/Tpl:</div><div>${escapeHtml([item.space, item.template].filter(Boolean).join(' / ') || '—')}</div>
            <div>Weights:</div><div>${escapeHtml(item.weights || item.weights_path || '—')}</div>
            <div>Lengths:</div><div>${escapeHtml(item.lengths || item.lengths_path || '—')}</div>
            <div>Mean/Median:</div><div>${(item.weights_mean!==undefined?Number(item.weights_mean).toFixed(3):'—')} / ${(item.weights_median!==undefined?Number(item.weights_median).toFixed(3):'—')}</div>
            <div>Min/Max:</div><div>${(item.weights_min!==undefined?Number(item.weights_min).toFixed(3):'—')} … ${(item.weights_max!==undefined?Number(item.weights_max).toFixed(3):'—')}</div>
          </div>
          <canvas id="histBig" style="width:100%; height:140px;"></canvas>
        </div>`;
      document.body.appendChild(modal);
      modal.style.display = 'flex';
      modal.querySelector('.backdrop').addEventListener('click', () => modal.remove());
      modal.querySelector('#closeSm').addEventListener('click', () => modal.remove());
      const histCanvas = modal.querySelector('#histBig');
      drawMiniHistogram(histCanvas, item.weights_histogram);
    }

    function applyCouplingPreset(item) {
      if (!item || typeof item !== 'object') return;
      // Name
      if (couplingNameInput) couplingNameInput.value = item.title || item.name || item.id || item.key || '';
      // Parameters
      cpRows.innerHTML = '';
      const params = extractParameters(item.parameters);
      if (params.length) {
        params.forEach(p => cpRows.appendChild(rowParam(p.name || '', valueToStr(p.value), p.unit || '', p.description || '')));
      } else {
        cpRows.appendChild(rowParam());
      }
      // Expressions
      const pre = item.pre_expression || item.pre || (item.equation && item.equation.pre_expression) || {};
      const post = item.post_expression || item.post || (item.equation && item.equation.post_expression) || {};
      const preL = section.querySelector('#couplingPreLhs');
      const preR = section.querySelector('#couplingPreRhs');
      const postL = section.querySelector('#couplingPostLhs');
      const postR = section.querySelector('#couplingPostRhs');
      if (preL) preL.value = pre.lhs || pre.label || 'pre';
      if (preR) preR.value = pre.rhs || pre.definition || '';
      if (postL) postL.value = post.lhs || post.label || 'post';
      if (postR) postR.value = post.rhs || post.definition || '';
      // Flags
      if (couplingSparseEl) couplingSparseEl.checked = !!item.sparse;
      if (couplingDelayedEl) couplingDelayedEl.checked = !!item.delayed;
      updateCouplingPreviews(section);
    }

    couplingSelect && couplingSelect.addEventListener('change', () => {
      const key = couplingSelect.value;
      const item = couplings.find(x => (x.key || x.id || x.name || x.title) == key);
      applyCouplingPreset(item);
  try { drawCouplingPlot(section); } catch(_) {}
    });

    const resetBtn = section.querySelector('#resetCouplingFromPreset');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        const key = couplingSelect.value;
        const item = couplings.find(x => (x.key || x.id || x.name || x.title) == key);
        applyCouplingPreset(item);
  try { drawCouplingPlot(section); } catch(_) {}
      });
    }

    // Initial auto-fill if a selection already exists
    function autoFillFromCurrentSelections() {
      if (modelSelect.value) {
        const evt = new Event('change');
        modelSelect.dispatchEvent(evt);
      }
      if (couplingSelect.value) {
        const evt = new Event('change');
        couplingSelect.dispatchEvent(evt);
      }
      if (networkSelect && networkSelect.value) {
        const evt = new Event('change');
        networkSelect.dispatchEvent(evt);
      }
    }
    autoFillFromCurrentSelections();
  try { drawCouplingPlot(section); } catch(_) {}

    // Spec generation
    const genBtn = section.querySelector('#generateSpec');
    genBtn && genBtn.addEventListener('click', () => {
      const spec = collectSpec(section, { models, networks, couplings });
      const yaml = toYAML(spec);
      const previewEl = section.querySelector('#specPreview');
      if (previewEl) previewEl.textContent = yaml;
    });

    // Spec preview actions: copy, download, expand
    const copySpecBtn = section.querySelector('#copySpecBtn');
    if (copySpecBtn) {
      copySpecBtn.addEventListener('click', () => {
        const spec = collectSpec(section, { models, networks, couplings });
        const yaml = toYAML(spec);
        copyText(yaml);
        toast('Spec YAML copied');
      });
    }
    const downloadSpecBtn = section.querySelector('#downloadSpecBtn');
    if (downloadSpecBtn) {
      downloadSpecBtn.addEventListener('click', () => {
        const spec = collectSpec(section, { models, networks, couplings });
        const yaml = toYAML(spec);
        const fname = (spec.label || spec.name || 'model_spec').toString().replace(/\s+/g,'_');
        downloadText(yaml, fname + '.yaml', 'text/yaml');
      });
    }
    const toggleSpecExpandBtn = section.querySelector('#toggleSpecExpand');
    const specPreviewEl = section.querySelector('#specPreview');
    if (toggleSpecExpandBtn && specPreviewEl) {
      toggleSpecExpandBtn.addEventListener('click', () => {
        specPreviewEl.classList.toggle('expanded');
        const expanded = specPreviewEl.classList.contains('expanded');
        toggleSpecExpandBtn.textContent = expanded ? 'Collapse' : 'Expand';
      });
    }

    // Download/Copy actions
    const modal = document.getElementById('builderModal');
    const copyModelBtn = modal.querySelector('#copyModelBtn');
    copyModelBtn && copyModelBtn.addEventListener('click', () => {
        const spec = collectSpec(section, { models, networks, couplings });
        const txt = toPythonLiteral(spec.model);
        copyText(txt);
        toast('Model copied');
      });
    const copyConnBtn = modal.querySelector('#copyConnBtn');
    copyConnBtn && copyConnBtn.addEventListener('click', () => {
        const spec = collectSpec(section, { models, networks, couplings });
        const txt = toPythonLiteral(spec.connectivity);
        copyText(txt);
        toast('Connectivity copied');
      });
    const copyCouplingBtn = modal.querySelector('#copyCouplingBtn');
    copyCouplingBtn && copyCouplingBtn.addEventListener('click', () => {
        const spec = collectSpec(section, { models, networks, couplings });
        const txt = toPythonLiteral(spec.coupling);
        copyText(txt);
        toast('Coupling copied');
      });
    const builderCopyBtn = modal.querySelector('#builderCopyPython');
    builderCopyBtn && builderCopyBtn.addEventListener('click', () => {
        const spec = collectSpec(section, { models, networks, couplings });
        const dict = toPythonLiteral(spec);
        const snippet = 'from tvbo import SimulationExperiment\n\n' +
          'SimulationExperiment(\n    **' + dict + '\n)\n';
        copyText(snippet);
        toast('Python snippet copied');
      });
  const builderDownloadBtn = modal.querySelector('#builderDownloadYaml');
  builderDownloadBtn && builderDownloadBtn.addEventListener('click', () => {
      const spec = collectSpec(section, { models, networks, couplings });
      const yaml = toYAML(spec);
  const fname = (spec.label || spec.name || 'model_spec').toString().replace(/\s+/g,'_');
  downloadText(yaml, fname + '.yaml', 'text/yaml');
    });
  }

  function extractParameters(paramsObj) {
    if (!paramsObj) return [];
    // Accept mapping or array of params
    if (Array.isArray(paramsObj)) return paramsObj.map(normalizeParam);
    if (paramsObj && typeof paramsObj === 'object') {
      return Object.entries(paramsObj).map(([k, v]) => normalizeParam({ name: k, ...(v || {}) }));
    }
    return [];
  }

  function normalizeParam(p) {
    const name = p?.name || '';
    const value = p?.value;
    const unit = p?.unit;
    const domain = p?.domain;
    const description = p?.description;
    return prune({ name, value, unit, domain, description });
  }

  function extractModelEquations(modelItem) {
    const eqs = [];
    const pushEq = (src) => {
      if (!src) return;
  const lhs = src.lhs || src.lefthandside || src.label;
  const rhs = src.rhs || src.righthandside || src.definition || src.pycode;
  if (lhs || rhs) eqs.push(prune({ lhs: lhs || undefined, rhs: rhs || undefined, pycode: src.pycode || rhs, latex: !!src.latex }));
    };
    // Handle both flat and nested under model
    const modelRoot = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;

    // derived_variables may be map or array
    const dv = modelRoot?.derived_variables;
    if (dv) {
      if (Array.isArray(dv)) {
        dv.forEach(d => pushEq(d?.equation || d));
      } else if (typeof dv === 'object') {
        Object.values(dv).forEach(d => pushEq(d?.equation || d));
      }
    }
    // state_variables equations
    const sv = modelRoot?.state_variables;
    if (sv) {
      if (Array.isArray(sv)) {
        sv.forEach(s => pushEq(s?.equation));
      } else if (typeof sv === 'object') {
        Object.values(sv).forEach(s => pushEq(s?.equation));
      }
    }
    // functions/output_transforms optional
    const fns = modelRoot?.functions;
    if (Array.isArray(fns)) {
      fns.forEach(fn => pushEq(fn?.equation || fn?.output));
    } else if (fns && typeof fns === 'object') {
      Object.values(fns).forEach(fn => pushEq(fn?.equation || fn?.output));
    }
    // derived_parameters equations (include as generic equations too)
    const dparams = modelRoot?.derived_parameters;
    if (dparams) {
      if (Array.isArray(dparams)) {
        dparams.forEach(dp => pushEq(dp?.equation));
      } else if (typeof dparams === 'object') {
        Object.values(dparams).forEach(dp => pushEq(dp?.equation));
      }
    }

    // Deep scan fallback: find any nested { equation: {lhs/rhs/...} }
    const seen = new Set();
    function scan(o) {
      if (!o || typeof o !== 'object') return;
      if (seen.has(o)) return; // avoid cycles
      seen.add(o);
      if (o.equation && typeof o.equation === 'object') {
        pushEq(o.equation);
      }
      for (const k of Object.keys(o)) {
        const v = o[k];
        if (v && typeof v === 'object') scan(v);
      }
    }
    scan(modelRoot);
    return eqs;
  }

  function extractFunctions(modelItem) {
    const out = [];
    const root = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;
    const fns = root?.functions;
    const push = (name, def) => {
      if (!def) return;
      const eq = def.equation || def;
      const rhs = eq?.rhs || eq?.definition || eq?.pycode || def?.output;
      const entry = prune({ name: name || def.name, rhs });
      if (entry) out.push(entry);
    };
    if (Array.isArray(fns)) fns.forEach(fn => push(fn?.name, fn));
    else if (fns && typeof fns === 'object') Object.entries(fns).forEach(([k, v]) => push(k, v));
    return out;
  }

  function extractStateEquations(modelItem) {
    const out = [];
    const root = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;
    const sv = root?.state_variables;
    const push = (def) => {
      if (!def) return;
      const eq = def.equation || def;
      if (!eq) return;
      const lhs = eq.lhs || eq.label;
      const rhs = eq.rhs || eq.definition || eq.pycode;
      const entry = prune({ lhs, rhs });
      if (entry) out.push(entry);
    };
    if (Array.isArray(sv)) sv.forEach(s => push(s));
    else if (sv && typeof sv === 'object') Object.values(sv).forEach(s => push(s));
    return out;
  }

  function extractDerivedVariables(modelItem) {
    const out = [];
    const root = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;
    const dv = root?.derived_variables;
    const push = (name, def) => {
      if (!def) return;
      const eq = def.equation || def;
      if (!eq) return;
      const lhs = eq.lhs || eq.label || name;
      const rhs = eq.rhs || eq.definition || eq.pycode;
      const entry = prune({ name: name || def.name, equation: prune({ lhs, rhs, latex: !!eq.latex }) });
      if (entry) out.push(entry);
    };
    if (Array.isArray(dv)) dv.forEach(d => push(d?.name, d));
    else if (dv && typeof dv === 'object') Object.entries(dv).forEach(([k, v]) => push(k, v));
    return out;
  }

  function extractOutputTransforms(modelItem) {
    const out = [];
    const root = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;
    const outs = root?.output_transforms;
    const push = (name, def) => {
      if (!def) return;
      const eq = def.equation || def;
      const rhs = eq?.rhs || eq?.definition || eq?.pycode || def?.output;
      const entry = prune({ name: name || def.name, equation: rhs ? prune({ rhs }) : undefined });
      if (entry) out.push(entry);
    };
    if (Array.isArray(outs)) outs.forEach(o => push(o?.name, o));
    else if (outs && typeof outs === 'object') Object.entries(outs).forEach(([k, v]) => push(k, v));
    return out;
  }

  function extractDerivedParameters(modelItem) {
    const out = [];
    const modelRoot = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;
    const dparams = modelRoot?.derived_parameters;
    const push = (name, def) => {
      if (!def) return;
      const eq = def.equation || def;
      if (!eq) return;
      const lhs = eq.lhs || eq.label || name;
      const rhs = eq.rhs || eq.definition || eq.pycode;
      const entry = prune({ name: name || def.name, equation: prune({ lhs, rhs, latex: !!eq.latex }) });
      if (entry) out.push(entry);
    };
    if (Array.isArray(dparams)) {
      dparams.forEach(dp => push(dp?.name, dp));
    } else if (dparams && typeof dparams === 'object') {
      Object.entries(dparams).forEach(([k, v]) => push(k, v));
    }
    return out;
  }

  function valueToStr(v) {
    if (v === undefined || v === null) return '';
    return typeof v === 'number' ? String(v) : v;
  }

  function collectSpec(section, lists) {
  const name = section.querySelector('#builderSpecName').value.trim() || 'MyNetworkModelSpec';
  const notesRaw = section.querySelector('#builderNotes').value;
  const notes = (notesRaw || '').trim();

    const modelKey = section.querySelector('#builderModel').value;
  const networkKey = section.querySelector('#builderNetwork')?.value;
    const couplingKey = section.querySelector('#builderCoupling').value;

    const modelItem = (lists.models || []).find(x => (x.key || x.id || x.name || x.title) == modelKey);
  const networkItem = (lists.networks || []).find(x => (x.key || x.id || x.name || x.title) == networkKey);
    const couplingItem = (lists.couplings || []).find(x => (x.key || x.id || x.name || x.title) == couplingKey);

    // Collect new section rows
    const dparamsUI = Array.from(section.querySelectorAll('#derivedParamsRows .builder-row')).map(row => ({
      name: row.querySelector('.eq-name').value.trim() || undefined,
      equation: prune({ lhs: row.querySelector('.eq-name').value.trim() || undefined, rhs: row.querySelector('.eq-expr').value.trim() || undefined })
    })).filter(e => e.name && (e.equation?.rhs || e.equation?.lhs));

    const functionsUI = Array.from(section.querySelectorAll('#functionsRows .builder-row')).map(row => ({
      name: row.querySelector('.eq-name').value.trim() || undefined,
      rhs: row.querySelector('.eq-expr').value.trim() || undefined,
    })).filter(f => f.name && f.rhs);

    const stateEqsUI = Array.from(section.querySelectorAll('#stateEqRows .builder-row')).map(row => ({
      lhs: row.querySelector('.eq-name').value.trim() || undefined,
      rhs: row.querySelector('.eq-expr').value.trim() || undefined,
    })).filter(e => e.lhs || e.rhs);

    const derivedVarsUI = Array.from(section.querySelectorAll('#derivedVarsRows .builder-row')).map(row => ({
      name: row.querySelector('.eq-name').value.trim() || undefined,
      equation: prune({ lhs: row.querySelector('.eq-name').value.trim() || undefined, rhs: row.querySelector('.eq-expr').value.trim() || undefined })
    })).filter(e => e.name && (e.equation?.rhs || e.equation?.lhs));

    const outputTransformsUI = Array.from(section.querySelectorAll('#outputTransformsRows .builder-row')).map(row => ({
      name: row.querySelector('.eq-name').value.trim() || undefined,
      equation: prune({ rhs: row.querySelector('.eq-expr').value.trim() || undefined })
    })).filter(e => e.name && e.equation?.rhs);

    const modelParams = Array.from(section.querySelectorAll('#modelParamsRows .builder-row')).map(row => ({
      name: row.querySelector('.p-name').value.trim(),
      value: parseMaybeNumber(row.querySelector('.p-value').value.trim()),
      unit: row.querySelector('.p-unit').value.trim() || undefined,
      description: row.nextElementSibling && row.nextElementSibling.classList && row.nextElementSibling.classList.contains('param-desc')
        ? (row.nextElementSibling.textContent || '').trim() || undefined
        : undefined,
    })).filter(p => p.name);

  const couplingParams = Array.from(section.querySelectorAll('#couplingParamsRows .builder-row')).map(row => ({
      name: row.querySelector('.p-name').value.trim(),
      value: parseMaybeNumber(row.querySelector('.p-value').value.trim()),
      unit: row.querySelector('.p-unit').value.trim() || undefined,
      description: row.nextElementSibling && row.nextElementSibling.classList && row.nextElementSibling.classList.contains('param-desc')
        ? (row.nextElementSibling.textContent || '').trim() || undefined
        : undefined,
    })).filter(p => p.name);

  // Coupling expressions and flags
  const couplingSparse = !!section.querySelector('#couplingSparse')?.checked;
  const preL = section.querySelector('#couplingPreLhs')?.value?.trim();
  const preR = section.querySelector('#couplingPreRhs')?.value?.trim();
  const postL = section.querySelector('#couplingPostLhs')?.value?.trim();
  const postR = section.querySelector('#couplingPostRhs')?.value?.trim();

    // Build connectivity depending on mode
    let connectome;
  const modeExisting = section.querySelector('#netModeExisting')?.checked;
  const modeToy = section.querySelector('#netModeToy')?.checked;
    if (modeExisting) {
      const tractSel = section.querySelector('#builderTract')?.value?.trim();
      const atlasSel = section.querySelector('#builderAtlas')?.value?.trim();
      let item = networkItem;
      if (!item && (tractSel || atlasSel)) {
        item = (lists.networks || []).find(n => {
          const tn = (n.tractogram || n.desc || '').toString().trim();
          const an = (n.atlas || n?.parcellation?.atlas?.name || '').toString().trim();
          return (!tractSel || tn === tractSel) && (!atlasSel || an === atlasSel);
        });
      }
      if (item) {
        connectome = prune({
          number_of_regions: item.n_regions || item.number_of_regions,
          parcellation: prune({
            label: item.parcellation?.label || undefined,
            atlas: prune({ name: item.atlas || item?.parcellation?.atlas?.name })
          }),
          tractogram: item.tractogram || item.desc || undefined,
          weights: prune({ dataLocation: item.weights_path || item.weights }),
          lengths: prune({ dataLocation: item.lengths_path || item.lengths }),
        });
      }
  } else if (modeToy) {
      const N = Math.max(2, Math.min(200, parseInt(section.querySelector('#toyN')?.value || '10', 10) || 10));
      const type = section.querySelector('#toyType')?.value || 'ring';
      const w = parseFloat(section.querySelector('#toyWeight')?.value || '1') || 1;
      const L = parseFloat(section.querySelector('#toyLength')?.value || '1') || 1;
      const p = parseFloat(section.querySelector('#toyP')?.value || '0.2') || 0.2;
      const selfLoops = !!section.querySelector('#toySelfLoops')?.checked;
      const weights = generateToyMatrix(N, type, w, selfLoops, p);
      const lengths = generateToyMatrix(N, type, L, selfLoops, p);
      const labels = Array.from({ length: N }, (_, i) => `R${i+1}`);
      connectome = prune({
        number_of_regions: N,
        parcellation: prune({ label: `toy-${type}-${N}`, atlas: prune({ name: 'toy' }) }),
        weights: { values: flattenMatrix(weights), x: { values: labels }, y: { values: labels } },
        lengths: { values: flattenMatrix(lengths), x: { values: labels }, y: { values: labels } },
      });
    }

  // Build top-level model using fetched full spec if present
  const full = STATE.lastFullModelSpec || modelItem || {};
  const baseModelName = (modelItem && (modelItem.title || modelItem.name || modelItem.id || modelItem.key)) || full.name || 'model';
  const derivedParamsOriginal = extractDerivedParameters(full);

  // Use explicit section collections instead of inferring from a single equations list
  const derivedParamsFromUI = dparamsUI;
  const derivedVarsFromUI = derivedVarsUI;

  const neuralMassModel = buildFullModelSpec(full, baseModelName, modelParams, derivedParamsOriginal, derivedVarsFromUI, derivedParamsFromUI, stateEqsUI, functionsUI, outputTransformsUI);

  const couplingName = section.querySelector('#builderCouplingName')?.value?.trim();
  const couplingDelayed = !!section.querySelector('#couplingDelayed')?.checked;
  const coupling = (couplingItem || couplingName || couplingParams.length || preL || preR || postL || postR || couplingSparse || couplingDelayed)
      ? prune({
      name: couplingName || (couplingItem ? (couplingItem.title || couplingItem.name || couplingItem.id || couplingItem.key) : undefined),
          parameters: couplingParams.length ? couplingParams : undefined,
          sparse: couplingSparse ? true : undefined,
      delayed: couplingDelayed ? true : undefined,
          pre_expression: (preL || preR) ? prune({ lhs: preL || undefined, rhs: preR || undefined }) : undefined,
          post_expression: (postL || postR) ? prune({ lhs: postL || undefined, rhs: postR || undefined }) : undefined,
        })
      : undefined;

  // Build SimulationExperiment at root (requested base class)
  const experiment = prune({
      id: 1,
      label: name,
      description: notes || undefined,
      model: neuralMassModel,
      connectivity: connectome,
      coupling: coupling,
      // equations are split into model.derived_variables and model.derived_parameters
    });

    log('Assembled SimulationExperiment object:', experiment);
    log('YAML preview:', '\n' + toYAML(experiment));
    return experiment;
  }

  const ALLOWED_MODEL_KEYS = new Set([
    'name','label','iri','has_reference','description',
    'parameters','derived_parameters','derived_variables','state_variables','functions','output_transforms','modes','stimulus','local_coupling_term','coupling_terms','number_of_modes','modified','derived_from_model'
  ]);

  function buildFullModelSpec(full, name, modelParams, derivedParamsOriginal, derivedVarsFromUI, derivedParamsFromUI, stateEqsUI, functionsUI, outputTransformsUI) {
    // Prefer the fetched full spec; filter only allowed model keys
    let base = {};
    let origDVShape = null; // 'object' | 'array' | null
    let origDPShape = null;
    if (full && typeof full === 'object') {
      // Some files might nest under 'model'
      const root = (full.model && typeof full.model === 'object') ? full.model : full;
      for (const k of Object.keys(root)) {
        if (ALLOWED_MODEL_KEYS.has(k)) base[k] = root[k];
      }
      if (root.derived_variables) origDVShape = Array.isArray(root.derived_variables) ? 'array' : (typeof root.derived_variables === 'object' ? 'object' : null);
      if (root.derived_parameters) origDPShape = Array.isArray(root.derived_parameters) ? 'array' : (typeof root.derived_parameters === 'object' ? 'object' : null);
    }
    base.name = name || base.name;
    if (modelParams && modelParams.length) base.parameters = modelParams;
    // State equations
    if (stateEqsUI && stateEqsUI.length) {
      base.state_variables = stateEqsUI.map(e => prune({ equation: prune({ lhs: e.lhs, rhs: e.rhs }) }));
    }
    // Merge derived parameters: original from model plus overrides from UI by name
    const mergedDerivedParams = mergeNamedList(toArray(base.derived_parameters || derivedParamsOriginal), toArray(derivedParamsFromUI));
    if (mergedDerivedParams.length) {
      base.derived_parameters = (origDPShape === 'object') ? toNamedObject(mergedDerivedParams) : mergedDerivedParams;
    }
    // Merge derived variables: original plus additions from UI (by name)
    const mergedDerivedVars = mergeNamedList(toArray(base.derived_variables), toArray(derivedVarsFromUI));
    if (mergedDerivedVars.length) {
      base.derived_variables = (origDVShape === 'object') ? toNamedObject(mergedDerivedVars) : mergedDerivedVars;
    }
    // Functions
    if (functionsUI && functionsUI.length) {
      base.functions = functionsUI.map(fn => prune({ name: fn.name, equation: prune({ rhs: fn.rhs }) }));
    }
    // Output transforms
    if (outputTransformsUI && outputTransformsUI.length) {
      base.output_transforms = outputTransformsUI.map(o => prune({ name: o.name, equation: prune({ rhs: o.equation?.rhs }) }));
    }
    // Reorder keys to canonical order for readability
    const ordered = {};
    const order = ['name','label','iri','has_reference','description','parameters','derived_parameters','functions','state_variables','derived_variables','output_transforms','modes','stimulus','local_coupling_term','coupling_terms','number_of_modes','modified','derived_from_model'];
    order.forEach(k => { if (base[k] !== undefined) ordered[k] = base[k]; });
    // include any remaining keys
    Object.keys(base).forEach(k => { if (ordered[k] === undefined) ordered[k] = base[k]; });
    return prune(ordered);
  }

  function toArray(x) { if (!x) return []; return Array.isArray(x) ? x : (typeof x==='object'? Object.values(x): [x]); }
  function mergeNamedList(originalList, overrideList) {
    const map = new Map();
    for (const it of (originalList||[])) { const n = it && (it.name || it.symbol); if (n) map.set(n, it); }
    for (const it of (overrideList||[])) { const n = it && (it.name || it.symbol); if (n) map.set(n, Object.assign({}, map.get(n)||{}, it)); }
    return Array.from(map.values());
  }
  function toNamedObject(list) {
    const obj = {};
    for (const it of (list||[])) { const n = it && (it.name || it.symbol); if (n) obj[n] = it; }
    return obj;
  }

  function extractParameterNamesFromModel(modelItem) {
    const names = new Set();
    const modelRoot = modelItem?.model && typeof modelItem.model === 'object' ? modelItem.model : modelItem;
    const addParamNames = (ps) => {
      if (!ps) return;
      if (Array.isArray(ps)) ps.forEach(p => p?.name && names.add(p.name));
      else if (typeof ps === 'object') Object.keys(ps).forEach(k => names.add(k));
    };
    addParamNames(modelRoot?.parameters);
    addParamNames(modelRoot?.derived_parameters);
    return names;
  }

  function guessModelJsonUrls(item) {
    const stems = [];
    const add = (s) => { if (s && !stems.includes(s)) stems.push(s); };
    add(item.key); add(item.id); add(item.name); add(item.title);
    const out = [];
    for (const s of stems) {
      const variants = [s, String(s).replace(/\s+/g,'_'), String(s).replace(/\s+/g,'')];
      for (const v of variants) {
        out.push(`models_json/${v}.json`);
      }
    }
    return out;
  }

  function pickItemRef(it) {
    const key = it.key || it.id || it.name || it.title;
    return prune({
      key,
      name: it.title || it.name || it.id || String(key),
      type: (it.type || '').toLowerCase() || undefined,
      source: it.source || it.url || it.href || undefined,
    });
  }

  function prune(obj) {
    if (Array.isArray(obj)) {
      return obj.map(prune).filter(v => v !== undefined);
    }
    if (obj && typeof obj === 'object') {
      const out = {};
      for (const k of Object.keys(obj)) {
        const v = prune(obj[k]);
        if (v !== undefined && !(Array.isArray(v) && v.length === 0)) out[k] = v;
      }
      return Object.keys(out).length ? out : undefined;
    }
    return obj === '' ? undefined : obj;
  }

  // Utilities for toy matrices
  function flattenMatrix(m) {
    if (!m || !Array.isArray(m)) return [];
    const out = [];
    for (const row of m) for (const v of row) out.push(v);
    return out;
  }
  function generateToyMatrix(N, type, value, selfLoops, p) {
    const A = Array.from({ length: N }, () => Array.from({ length: N }, () => 0));
    const set = (i, j) => { if (i === j && !selfLoops) return; A[i][j] = value; };
    if (type === 'complete') {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) set(i, j);
    } else if (type === 'line') {
      for (let i = 0; i < N - 1; i++) { set(i, i + 1); set(i + 1, i); }
    } else if (type === 'star') {
      for (let j = 1; j < N; j++) { set(0, j); set(j, 0); }
    } else if (type === 'erdos') {
      const prob = Math.max(0, Math.min(1, p || 0));
      for (let i = 0; i < N; i++) {
        for (let j = i + 1; j < N; j++) {
          if (Math.random() < prob) { set(i, j); set(j, i); }
        }
      }
    } else { // ring default
      for (let i = 0; i < N; i++) { const j = (i + 1) % N; set(i, j); set(j, i); }
    }
    return A;
  }

  function parseMaybeNumber(v) {
    if (v === '') return undefined;
    const n = Number(v);
    return Number.isFinite(n) && String(n) === v ? n : v;
  }

  // Draw a tiny sampled coupling curve gx -> post using current params and pre/post expressions
  function drawCouplingPlot(section) {
    const canvas = section.querySelector('#couplingPlot');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width = canvas.clientWidth || 300;
    const h = canvas.height = 140;
    ctx.clearRect(0,0,w,h);

    // Collect parameters
    const params = {};
    section.querySelectorAll('#couplingParamsRows .builder-row').forEach(row => {
      const name = row.querySelector('.p-name')?.value?.trim();
      const valRaw = row.querySelector('.p-value')?.value?.trim();
      if (name && valRaw !== '') {
        const v = Number(valRaw);
        params[name] = Number.isFinite(v) ? v : 0;
      }
    });
    const preExpr = section.querySelector('#couplingPreRhs')?.value?.trim();
    const postExpr = section.querySelector('#couplingPostRhs')?.value?.trim();
    if (!postExpr) return; // need at least post
    const xmin = Number(section.querySelector('#gxMin')?.value || -2);
    const xmax = Number(section.querySelector('#gxMax')?.value || 2);
    const N = Math.max(20, Math.floor((w - 40) / 4));
    const xs = Array.from({length: N}, (_,i)=> xmin + (i*(xmax-xmin)/(N-1)));

    function evalExpr(expr, scope) {
      if (!expr) return undefined;
      let code = expr.replace(/\^/g,'**');
      const names = ['sin','cos','tan','tanh','log','sqrt','exp'];
      names.forEach(fn => { const r = new RegExp(`\\b${fn}\\b`,'g'); code = code.replace(r, `Math.${fn}`); });
      const vars = Object.keys(scope || {});
      const vals = vars.map(k => scope[k]);
      try {
        // eslint-disable-next-line no-new-func
        const f = new Function(...vars, `return (${code});`);
        return f(...vals);
      } catch { return undefined; }
    }

    const ys = [];
    const CLAMP = 1e6;
    xs.forEach(gx => {
      let pre = gx;
      // Treat x_j and x_i as small vectors to support indexing like x_j[0]-x_j[1]
      const xj = [gx, 0, 0, 0];
      const xi = [0, 0, 0, 0];
      if (preExpr) {
        const env = Object.assign({}, params, { gx, x_j: xj, x_i: xi });
        const val = evalExpr(preExpr, env);
        if (typeof val === 'number' && isFinite(val)) pre = val;
      }
      const postEnv = Object.assign({}, params, { gx: pre, x_j: xj, x_i: xi });
      let out = evalExpr(postExpr, postEnv);
      if (typeof out === 'number' && isFinite(out)) {
        if (Math.abs(out) > CLAMP) out = Math.sign(out) * CLAMP;
        ys.push(out);
      } else {
        ys.push(NaN);
      }
    });

    // Axes
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(30, h-18); ctx.lineTo(w-6, h-18); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(30, 6); ctx.lineTo(30, h-18); ctx.stroke();

    const ymin = Math.min(...ys.filter(v => isFinite(v)), 0);
    const ymax = Math.max(...ys.filter(v => isFinite(v)), 0.001);
    const xToPx = (x)=> 30 + (x - xmin) * (w-36) / (xmax - xmin || 1);
    const yToPx = (y)=> (h-18) - (y - ymin) * (h-24) / (ymax - ymin || 1);

    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    xs.forEach((x, i) => {
      const y = ys[i];
      const px = xToPx(x);
      const py = yToPx(isFinite(y) ? y : 0);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  // YAML emitter focusing on readability and proper indentation/newlines
  function toYAML(obj, indent = 0) {
    const pad = '  '.repeat(indent);
    if (obj === null || obj === undefined) return '';
    if (typeof obj === 'string') return safeScalar(obj);
    if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
    if (Array.isArray(obj)) {
      if (obj.length === 0) return '[]';
      return obj.map(v => {
        if (v && typeof v === 'object' && !Array.isArray(v)) {
          // Inline first line of object after '- ' and keep remaining lines properly indented
          const inner = toYAML(v, indent + 1);
          const lines = inner.split('\n');
          const first = lines[0] ? lines[0].replace(new RegExp(`^${pad}  `), '') : '';
          const rest = lines.slice(1).join('\n');
          return `${pad}- ${first}${rest ? '\n' + rest : ''}`;
        }
        return `${pad}- ${asBlock(v, indent + 1, true)}`;
      }).join('\n');
    }
    if (typeof obj === 'object') {
      const entries = Object.entries(obj).filter(([, v]) => v !== undefined && !(Array.isArray(v) && v.length === 0));
      const parts = entries.map(([k, v]) => {
        const key = safeKey(k);
        if (v && typeof v === 'object') {
          if (Array.isArray(v)) {
            const arr = toYAML(v, indent + 1);
            const arrLines = arr.split('\n').map(l => (l ? '  ' + l : l)).join('\n');
            if (arr === '[]') return `${pad}${key}: []`;
            return `${pad}${key}:\n${arrLines}`;
          } else {
            const nested = toYAML(v, indent + 1);
            const lines = nested.split('\n').map(l => (l ? '  ' + l : l)).join('\n');
            return `${pad}${key}:\n${lines}`;
          }
        }
        return `${pad}${key}: ${toYAML(v, indent)}`;
      });
      // Insert a blank line between top-level major blocks for readability
      if (indent === 0) {
        return parts.join('\n\n');
      }
      return parts.join('\n');
    }
    return '';
  }

  function asBlock(v, indent, inArray) {
    if (v && typeof v === 'object') {
      const inner = toYAML(v, indent);
      if (!inner) return '{}';
      const lines = inner.split('\n');
      if (inArray && lines.length > 1) {
        // first line continues after "- ", subsequent are indented
        return `\n${lines.map(l => '  ' + l).join('\n')}`;
      }
      return inner;
    }
    return toYAML(v, indent - 1);
  }

  function safeKey(k) {
    return /[:\s]/.test(k) ? JSON.stringify(k) : k;
  }
  function safeScalar(s) {
    if (s === '' || /[:#\-?&*!|>'"%@`{}\[\],]|^\s|\s$|\n/.test(s)) {
      return JSON.stringify(s);
    }
    return s;
  }

  function escapeHtml(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }
  function escapeAttr(s) {
    return String(s || '')
      .replace(/"/g, '&quot;');
  }

  // Render LaTeX preview for an equation row using MathJax
  function updateEquationPreview(row) {
    const name = row.querySelector('.eq-name')?.value?.trim();
    const expr = row.querySelector('.eq-expr')?.value?.trim();
    const preview = row.querySelector('.eq-preview');
    if (!preview) return;
    if (!STATE.previewEnabled) {
      preview.style.display = 'none';
      preview.innerHTML = '';
      return;
    }
    preview.style.display = 'block';
    if (!name && !expr) { preview.innerHTML = ''; return; }
    const tex = toTex(name, expr);
    preview.innerHTML = tex ? `$${tex}$` : '';
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([preview]).catch(() => {
        // Fallback to plain text on MathJax error
        preview.textContent = tex;
        preview.style.fontFamily = 'monospace';
        preview.style.color = '#374151';
      });
      // Style rendered SVG to appear smaller
      setTimeout(() => {
        const svg = preview.querySelector('svg');
        if (svg) {
          svg.style.maxHeight = '1.2em';
        }
      }, 0);
    }
  }

  // Render LaTeX preview for coupling expressions using existing toTex
  function updateCouplingPreviews(section) {
    const enabled = !!section.querySelector('#toggleCouplingEqPreview')?.checked;
    const prePrev = section.querySelector('#couplingPrePreview');
    const postPrev = section.querySelector('#couplingPostPreview');
    const preL = section.querySelector('#couplingPreLhs')?.value?.trim();
    const preR = section.querySelector('#couplingPreRhs')?.value?.trim();
    const postL = section.querySelector('#couplingPostLhs')?.value?.trim();
    const postR = section.querySelector('#couplingPostRhs')?.value?.trim();
    const apply = (el, lhs, rhs) => {
      if (!el) return;
      if (!enabled) { el.style.display = 'none'; el.innerHTML = ''; return; }
      el.style.display = 'block';
      const tex = toTex(lhs, rhs);
      el.innerHTML = tex ? `$${tex}$` : '';
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([el]).catch(() => {
          el.textContent = tex;
          el.style.fontFamily = 'monospace';
          el.style.color = '#374151';
        });
        setTimeout(() => { const svg = el.querySelector('svg'); if (svg) svg.style.maxHeight = '1.2em'; }, 0);
      }
    };
    apply(prePrev, preL, preR);
    apply(postPrev, postL, postR);
  }

  // Minimal Python-like to LaTeX conversion for nice previews
  function toTex(lhs, rhs) {
    let L = lhs || '';
    let R = rhs || '';
    if (!L && !R) return '';
    // Basic power operator
    R = R.replace(/\*\*/g, '^');
    // exp(arg) -> e^{arg}
    R = R.replace(/\bexp\s*\(([^()]*)\)/g, (m, a) => `e^{${a}}`);
    // sqrt(arg) -> \sqrt{arg}
    R = R.replace(/\bsqrt\s*\(([^()]*)\)/g, (m, a) => `\\sqrt{${a}}`);
    // Trig/log functions: sin -> \sin, etc. Keep parentheses as-is
    R = R.replace(/\b(sin|cos|tan|tanh|log)\b/g, (m, fn) => `\\${fn}`);
    // Escape underscores to avoid MathJax subscript parsing issues
    L = L.replace(/_/g, '\\_');
    R = R.replace(/_/g, '\\_');
    // Insert \cdot for * between symbols/numbers (modern browsers support lookbehind)
    try {
      R = R.replace(/(?<=\b[\w)\}])\*(?=[\w(\{\\])/g, ' \\cdot ');
    } catch { // fallback without lookbehind
      R = R.replace(/([0-9A-Za-z_\)\}])\*([0-9A-Za-z_\\\(\{])/g, '$1 \\cdot $2');
    }
    const lhsTex = L;
    const rhsTex = R;
    return lhsTex && rhsTex ? `${lhsTex} = ${rhsTex}` : (lhsTex || rhsTex);
  }

  function copyText(text) {
    navigator.clipboard?.writeText(text).catch(() => {
      const ta = document.createElement('textarea');
      ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    });
  }

  function downloadText(text, filename, mime = 'text/plain') {
    const blob = new Blob([text], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 500);
  }

  function toast(msg) {
    const el = document.createElement('div');
    el.textContent = msg;
    el.style.position = 'fixed';
    el.style.bottom = '20px';
    el.style.right = '20px';
    el.style.background = '#1f2937';
    el.style.color = 'white';
    el.style.padding = '8px 12px';
    el.style.borderRadius = '6px';
    el.style.boxShadow = '0 6px 18px rgba(0,0,0,.2)';
    el.style.zIndex = '2000';
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2000);
  }

  // Init
  document.addEventListener('DOMContentLoaded', () => {
    whenDataReady(() => {
      createButton();
    });
  });

  // Serialize a JS object to a Python dict literal with True/False/None
  function toPythonLiteral(obj, indent = 0) {
    const pad = '  '.repeat(indent);
    if (obj === null || obj === undefined) return 'None';
    if (typeof obj === 'number') return String(obj);
    if (typeof obj === 'boolean') return obj ? 'True' : 'False';
    if (typeof obj === 'string') return JSON.stringify(obj);
    if (Array.isArray(obj)) {
      if (obj.length === 0) return '[]';
      const inner = obj.map(v => toPythonLiteral(v, indent + 1)).join(', ');
      // pretty if long
      if (inner.length > 80 || obj.some(v => typeof v === 'object')) {
        const lines = obj.map(v => pad + '  ' + toPythonLiteral(v, indent + 1));
        return '[\n' + lines.join(',\n') + '\n' + pad + ']';
      }
      return '[' + inner + ']';
    }
    if (typeof obj === 'object') {
      const entries = Object.entries(obj).filter(([,v]) => v !== undefined);
      if (entries.length === 0) return '{}';
      const parts = entries.map(([k, v]) => `${JSON.stringify(k)}: ${toPythonLiteral(v, indent + 1)}`);
      const oneLine = parts.join(', ');
      if (oneLine.length <= 100 && !entries.some(([,v]) => typeof v === 'object')) {
        return '{ ' + oneLine + ' }';
      }
      const lines = entries.map(([k, v]) => pad + '  ' + JSON.stringify(k) + ': ' + toPythonLiteral(v, indent + 1));
      return '{\n' + lines.join(',\n') + '\n' + pad + '}';
    }
    return 'None';
  }

  // Expose functions globally for external use
  window.openBuilderModal = openBuilderModal;
  window.closeBuilderModal = closeBuilderModal;
})();
