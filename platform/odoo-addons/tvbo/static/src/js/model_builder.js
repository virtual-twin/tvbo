// Simplified Model Builder for Odoo TVBO Configurator
// Adapted from browser model_builder.js

(function () {
  const STATE = {
    dataLoaded: false,
    data: [],
    lastFullModelSpec: null,
    previewEnabled: false,
  };

  const DEBUG = true;
  const log = (...args) => { if (DEBUG && console && console.log) console.log('[ModelBuilder]', ...args); };

  // Global error handler to help debug issues
  window.addEventListener('error', function(event) {
    console.error('[ModelBuilder] Global error caught:', event.message, 'at', event.filename, ':', event.lineno);
  });

  function initializeBuilder() {
    try {
      // Load data
      STATE.data = window.searchData || [];
      STATE.dataLoaded = true;
      log('Initializing builder with data:', { count: STATE.data.length });

      const content = document.getElementById('builderContent');
      if (content) {
        renderBuilder(content);
      }
    } catch (err) {
      console.error('[ModelBuilder] Error in initializeBuilder:', err);
    }
  }

  function renderBuilder(root) {
    const data = STATE.data || [];
    log('Rendering builder with data count:', data.length);
    const models = data.filter(x => (x.type || '').toLowerCase() === 'model');
    log('Filtered models count:', models.length);

    root.innerHTML = `
      <div>
        <div class="builder-field" style="display: grid; grid-template-columns: 2fr 1fr; gap: 16px; align-items: end;">
          <div>
            <label>Model Name</label>
            <input id="builderSpecName" class="builder-input" placeholder="MyCustomModel" />
          </div>
          <div>
            <label>System Type</label>
            <select id="builderSystemType" class="builder-select">
              <option value="continuous">Continuous (ODE/SDE)</option>
              <option value="discrete">Discrete (Maps)</option>
            </select>
          </div>
        </div>
        <div class="builder-field">
          <label>Description</label>
          <textarea id="builderNotes" class="builder-text" rows="2" placeholder="Optional description"></textarea>
        </div>
      </div>

      <div class="hr"></div>

      <div>
        <div class="builder-field">
          <div class="builder-subtitle">Base Model (optional)</div>
          <select id="builderModel" class="builder-select">
            <option value="">— select base model or build from scratch —</option>
            ${models.map(m => `<option value="${escapeHtml(m.key || m.id || m.name)}">${escapeHtml(m.title || m.name)}</option>`).join('')}
          </select>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle" style="display:flex; align-items:center; justify-content:space-between;">
            <span>Parameters</span>
            <label style="display:flex; align-items:center; gap:6px; font-size: 0.9em; color:#4a5568;">
              <input type="checkbox" id="toggleEqPreview" /> Show LaTeX preview
            </label>
          </div>
          <div id="modelParamsRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addModelParam">Add parameter</button>
          </div>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle">Derived Parameters</div>
          <div id="derivedParamsRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addDerivedParam">Add derived parameter</button>
          </div>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle">State Variables</div>
          <div id="stateEqRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addStateEquation">Add state variable</button>
          </div>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle">Derived Variables</div>
          <div id="derivedVarsRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addDerivedVariable">Add derived variable</button>
          </div>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle">Output Transforms</div>
          <div id="outputTransformsRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addOutputTransform">Add output transform</button>
          </div>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle">Functions</div>
          <div id="functionsRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addFunction">Add function</button>
          </div>
        </div>

        <div class="builder-field">
          <div class="builder-subtitle">Coupling Terms</div>
          <div id="couplingTermsRows" class="builder-rows"></div>
          <div class="builder-actions">
            <button class="btn btn-sm btn-secondary" id="addCouplingTerm">Add coupling term</button>
          </div>
        </div>
      </div>

      <div class="hr"></div>

      <div style="display:flex; gap:8px; align-items:center; flex-wrap: wrap;">
        <button class="btn btn-primary" id="generateSpec">Generate Preview</button>
      </div>
      <div id="specPreview" class="preview" style="margin-top:8px; white-space: pre-wrap; font-family: monospace; font-size: 12.5px; line-height: 1.35; overflow: auto;"></div>
    `;

    const mpRows = root.querySelector('#modelParamsRows');
    const dpRows = root.querySelector('#derivedParamsRows');
    const seRows = root.querySelector('#stateEqRows');
    const dvRows = root.querySelector('#derivedVarsRows');
    const otRows = root.querySelector('#outputTransformsRows');
    const fnRows = root.querySelector('#functionsRows');
    const ctRows = root.querySelector('#couplingTermsRows');

    // Row creation helpers
    function rowParam(name = '', value = '', unit = '', symbol = '', domain_lo = '', domain_hi = '') {
      const div = document.createElement('div');
      div.className = 'builder-row';
      div.style.gridTemplateColumns = '1fr 1fr 0.8fr 0.8fr 0.8fr 0.8fr auto';
      div.innerHTML = `
        <input class="builder-input p-name" placeholder="name" value="${escapeAttr(name)}" />
        <input class="builder-input p-value" placeholder="value" value="${escapeAttr(value)}" />
        <input class="builder-input p-unit" placeholder="unit" value="${escapeAttr(unit)}" />
        <input class="builder-input p-symbol" placeholder="symbol" value="${escapeAttr(symbol)}" />
        <input class="builder-input p-domain-lo" placeholder="min" value="${escapeAttr(domain_lo)}" />
        <input class="builder-input p-domain-hi" placeholder="max" value="${escapeAttr(domain_hi)}" />
        <button class="btn btn-sm btn-danger p-del" title="Remove">✕</button>`;
      div.querySelector('.p-del').addEventListener('click', () => div.remove());
      return div;
    }

    function rowDerivedParam(name = '', expr = '', unit = '') {
      const div = document.createElement('div');
      div.className = 'builder-row';
      div.style.gridTemplateColumns = '1fr 2fr 1fr auto';
      div.innerHTML = `
        <input class="builder-input dp-name" placeholder="name" value="${escapeAttr(name)}" />
        <input class="builder-input dp-expr" placeholder="expression" value="${escapeAttr(expr)}" />
        <input class="builder-input dp-unit" placeholder="unit" value="${escapeAttr(unit)}" />
        <button class="btn btn-sm btn-danger dp-del" title="Remove">✕</button>
        <div class="eq-preview" style="grid-column: 1 / 4; display:none; font-size: 0.9em; color:#1f2937;"></div>`;
      div.querySelector('.dp-del').addEventListener('click', () => div.remove());
      const update = () => updateEquationPreview(div);
      div.querySelector('.dp-name').addEventListener('input', update);
      div.querySelector('.dp-expr').addEventListener('input', update);
      update();
      return div;
    }

    function rowStateVar(name = '', expr = '', symbol = '', unit = '', initial = '0.1', voi = true, coupling = false) {
      const div = document.createElement('div');
      div.className = 'builder-row';
      div.style.gridTemplateColumns = '1fr 2fr 0.6fr 0.8fr 0.8fr auto auto auto';
      div.innerHTML = `
        <input class="builder-input sv-name" placeholder="name" value="${escapeAttr(name)}" />
        <input class="builder-input sv-expr" placeholder="d/dt expression" value="${escapeAttr(expr)}" />
        <input class="builder-input sv-symbol" placeholder="symbol" value="${escapeAttr(symbol)}" />
        <input class="builder-input sv-unit" placeholder="unit" value="${escapeAttr(unit)}" />
        <input class="builder-input sv-initial" placeholder="initial" value="${escapeAttr(initial)}" />
        <label class="sv-voi-label"><input type="checkbox" class="sv-voi" ${voi ? 'checked' : ''} /> VOI</label>
        <label class="sv-coupling-label"><input type="checkbox" class="sv-coupling" ${coupling ? 'checked' : ''} /> Coupling</label>
        <button class="btn btn-sm btn-danger sv-del" title="Remove">✕</button>
        <div class="eq-preview" style="grid-column: 1 / 8; display:none; font-size: 0.9em; color:#1f2937;"></div>`;
      div.querySelector('.sv-del').addEventListener('click', () => div.remove());
      const update = () => updateEquationPreview(div);
      div.querySelector('.sv-name').addEventListener('input', update);
      div.querySelector('.sv-expr').addEventListener('input', update);
      update();
      return div;
    }

    function rowEquation(name = '', expr = '', unit = '', className = 'eq') {
      const div = document.createElement('div');
      div.className = 'builder-row';
      div.style.gridTemplateColumns = '1fr 2fr 1fr auto';
      div.innerHTML = `
        <input class="builder-input ${className}-name" placeholder="name" value="${escapeAttr(name)}" />
        <input class="builder-input ${className}-expr" placeholder="expression" value="${escapeAttr(expr)}" />
        <input class="builder-input ${className}-unit" placeholder="unit" value="${escapeAttr(unit)}" />
        <button class="btn btn-sm btn-danger ${className}-del" title="Remove">✕</button>
        <div class="eq-preview" style="grid-column: 1 / 4; display:none; font-size: 0.9em; color:#1f2937;"></div>`;
      div.querySelector(`.${className}-del`).addEventListener('click', () => div.remove());
      const update = () => updateEquationPreview(div);
      div.querySelector(`.${className}-name`).addEventListener('input', update);
      div.querySelector(`.${className}-expr`).addEventListener('input', update);
      update();
      return div;
    }

    function rowFunction(name = '', expr = '') {
      const div = document.createElement('div');
      div.className = 'builder-row';
      div.style.gridTemplateColumns = '1fr 2fr auto';
      div.innerHTML = `
        <input class="builder-input fn-name" placeholder="function name" value="${escapeAttr(name)}" />
        <input class="builder-input fn-expr" placeholder="expression" value="${escapeAttr(expr)}" />
        <button class="btn btn-sm btn-danger fn-del" title="Remove">✕</button>
        <div class="eq-preview" style="grid-column: 1 / 3; display:none; font-size: 0.9em; color:#1f2937;"></div>`;
      div.querySelector('.fn-del').addEventListener('click', () => div.remove());
      const update = () => updateEquationPreview(div);
      div.querySelector('.fn-name').addEventListener('input', update);
      div.querySelector('.fn-expr').addEventListener('input', update);
      update();
      return div;
    }

    function rowCouplingTerm(name = '', value = '0.0') {
      const div = document.createElement('div');
      div.className = 'builder-row';
      div.style.gridTemplateColumns = '1fr 1fr auto';
      div.innerHTML = `
        <input class="builder-input ct-name" placeholder="coupling term name" value="${escapeAttr(name)}" />
        <input class="builder-input ct-value" placeholder="default value" value="${escapeAttr(value)}" />
        <button class="btn btn-sm btn-danger ct-del" title="Remove">✕</button>`;
      div.querySelector('.ct-del').addEventListener('click', () => div.remove());
      return div;
    }

    // Initial rows - commented out to show only "Add" buttons by default
    // mpRows.appendChild(rowParam());
    // dpRows.appendChild(rowDerivedParam());
    // seRows.appendChild(rowStateVar());
    // dvRows.appendChild(rowEquation('', '', '', 'dv'));
    // otRows.appendChild(rowEquation('', '', '', 'ot'));
    // fnRows.appendChild(rowFunction());
    // ctRows.appendChild(rowCouplingTerm());

    // Add buttons
    root.querySelector('#addModelParam').addEventListener('click', () => mpRows.appendChild(rowParam()));
    root.querySelector('#addDerivedParam').addEventListener('click', () => dpRows.appendChild(rowDerivedParam()));
    root.querySelector('#addStateEquation').addEventListener('click', () => seRows.appendChild(rowStateVar()));
    root.querySelector('#addDerivedVariable').addEventListener('click', () => dvRows.appendChild(rowEquation('', '', '', 'dv')));
    root.querySelector('#addOutputTransform').addEventListener('click', () => otRows.appendChild(rowEquation('', '', '', 'ot')));
    root.querySelector('#addFunction').addEventListener('click', () => fnRows.appendChild(rowFunction()));
    root.querySelector('#addCouplingTerm').addEventListener('click', () => ctRows.appendChild(rowCouplingTerm()));

    // LaTeX preview toggle
    const previewToggle = root.querySelector('#toggleEqPreview');
    if (previewToggle) {
      previewToggle.checked = STATE.previewEnabled;
      previewToggle.addEventListener('change', () => {
        STATE.previewEnabled = !!previewToggle.checked;
        root.querySelectorAll('.builder-row').forEach(row => {
          const prev = row.querySelector('.eq-preview');
          if (prev) {
            prev.style.display = STATE.previewEnabled ? 'block' : 'none';
            updateEquationPreview(row);
          }
        });
      });
    }

    // Model selection auto-fill
    const modelSelect = root.querySelector('#builderModel');
    modelSelect && modelSelect.addEventListener('change', async () => {
      const key = modelSelect.value;
      log('Model selected:', key);
      const item = models.find(x => (x.key || x.id || x.name) == key);

      mpRows.innerHTML = '';
      dpRows.innerHTML = '';
      seRows.innerHTML = '';
      dvRows.innerHTML = '';
      otRows.innerHTML = '';
      fnRows.innerHTML = '';
      ctRows.innerHTML = '';

      if (!item) {
        // Don't add empty rows - let user add them with buttons
        return;
      }

      STATE.lastFullModelSpec = item;
      log('Loading model data:', item);

      // Fill system type
      const systemTypeSelect = root.querySelector('#builderSystemType');
      if (systemTypeSelect && item.system_type) {
        systemTypeSelect.value = item.system_type;
      }

      // Fill parameters
      let params = item.parameters || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(params) && typeof params === 'object') {
        params = Object.keys(params).map(key => ({
          name: key,
          ...params[key]
        }));
      }
      if (params.length) {
        params.forEach(p => {
          const domain_lo = p.domain?.lo !== undefined ? p.domain.lo : '';
          const domain_hi = p.domain?.hi !== undefined ? p.domain.hi : '';
          mpRows.appendChild(rowParam(p.name || '', valueToStr(p.value), p.unit || '', p.symbol || '', domain_lo, domain_hi));
        });
      }

      // Fill derived parameters
      let dparams = item.derived_parameters || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(dparams) && typeof dparams === 'object') {
        dparams = Object.keys(dparams).map(key => ({
          name: key,
          ...dparams[key]
        }));
      }
      if (dparams.length) {
        dparams.forEach(dp => {
          const eq = dp.equation || {};
          dpRows.appendChild(rowDerivedParam(dp.name || '', eq.rhs || '', dp.unit || ''));
        });
      }

      // Fill state variables
      let svs = item.state_variables || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(svs) && typeof svs === 'object') {
        svs = Object.keys(svs).map(key => ({
          name: key,
          ...svs[key]
        }));
      }
      if (svs.length) {
        svs.forEach(sv => {
          const eq = sv.equation || {};
          const voi = sv.variable_of_interest !== undefined ? sv.variable_of_interest : true;
          const coupling = sv.coupling_variable || false;
          seRows.appendChild(rowStateVar(
            sv.name || '',
            eq.rhs || '',
            sv.symbol || '',
            sv.unit || '',
            valueToStr(sv.initial_value !== undefined ? sv.initial_value : 0.1),
            voi,
            coupling
          ));
        });
      }

      // Fill derived variables
      let dvs = item.derived_variables || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(dvs) && typeof dvs === 'object') {
        dvs = Object.keys(dvs).map(key => ({
          name: key,
          ...dvs[key]
        }));
      }
      if (dvs.length) {
        dvs.forEach(dv => {
          const eq = dv.equation || {};
          dvRows.appendChild(rowEquation(dv.name || '', eq.rhs || '', dv.unit || '', 'dv'));
        });
      }

      // Fill output transforms
      let ots = item.output || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(ots) && typeof ots === 'object') {
        ots = Object.keys(ots).map(key => ({
          name: key,
          ...ots[key]
        }));
      }
      if (ots.length) {
        ots.forEach(ot => {
          const eq = ot.equation || {};
          otRows.appendChild(rowEquation(ot.name || '', eq.rhs || '', ot.unit || '', 'ot'));
        });
      }

      // Fill functions
      let fns = item.functions || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(fns) && typeof fns === 'object') {
        fns = Object.keys(fns).map(key => ({
          name: key,
          ...fns[key]
        }));
      }
      if (fns.length) {
        fns.forEach(fn => {
          const eq = fn.equation || {};
          fnRows.appendChild(rowFunction(fn.name || '', eq.rhs || eq.definition || ''));
        });
      }

      // Fill coupling terms
      let cts = item.coupling_terms || [];
      // Handle both array and object formats (YAML can be dict)
      if (!Array.isArray(cts) && typeof cts === 'object') {
        cts = Object.keys(cts).map(key => ({
          name: key,
          value: 0.0,
          ...cts[key]
        }));
      }
      if (cts.length) {
        cts.forEach(ct => {
          ctRows.appendChild(rowCouplingTerm(ct.name || '', valueToStr(ct.value !== undefined ? ct.value : 0.0)));
        });
      }
    });

    // Generate preview
    root.querySelector('#generateSpec').addEventListener('click', () => {
      const spec = collectSpec(root, { models });
      const yaml = generateYamlContent(spec);
      root.querySelector('#specPreview').textContent = yaml;
    });
  }

  function updateEquationPreview(row) {
    // Determine the type of row and get appropriate inputs
    let name, expr;

    if (row.querySelector('.sv-name')) {
      // State variable row
      name = row.querySelector('.sv-name')?.value?.trim();
      expr = row.querySelector('.sv-expr')?.value?.trim();
    } else if (row.querySelector('.dp-name')) {
      // Derived parameter row
      name = row.querySelector('.dp-name')?.value?.trim();
      expr = row.querySelector('.dp-expr')?.value?.trim();
    } else if (row.querySelector('.dv-name')) {
      // Derived variable row
      name = row.querySelector('.dv-name')?.value?.trim();
      expr = row.querySelector('.dv-expr')?.value?.trim();
    } else if (row.querySelector('.ot-name')) {
      // Output transform row
      name = row.querySelector('.ot-name')?.value?.trim();
      expr = row.querySelector('.ot-expr')?.value?.trim();
    } else if (row.querySelector('.fn-name')) {
      // Function row
      name = row.querySelector('.fn-name')?.value?.trim();
      expr = row.querySelector('.fn-expr')?.value?.trim();
    } else {
      // Fallback to old names
      name = row.querySelector('.eq-name')?.value?.trim();
      expr = row.querySelector('.eq-expr')?.value?.trim();
    }

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
        preview.textContent = tex;
        preview.style.fontFamily = 'monospace';
      });
    }
  }

  function collectSpec(section, lists) {
    const name = section.querySelector('#builderSpecName')?.value?.trim() || 'MyCustomModel';
    const description = section.querySelector('#builderNotes')?.value?.trim() || '';
    const systemType = section.querySelector('#builderSystemType')?.value || 'continuous';

    const modelParams = Array.from(section.querySelectorAll('#modelParamsRows .builder-row')).map(row => {
      const domain_lo = parseMaybeNumber(row.querySelector('.p-domain-lo')?.value?.trim());
      const domain_hi = parseMaybeNumber(row.querySelector('.p-domain-hi')?.value?.trim());
      const domain = (domain_lo !== undefined || domain_hi !== undefined) ? { lo: domain_lo, hi: domain_hi } : undefined;

      return {
        name: row.querySelector('.p-name')?.value?.trim() || '',
        value: parseMaybeNumber(row.querySelector('.p-value')?.value?.trim()),
        unit: row.querySelector('.p-unit')?.value?.trim() || undefined,
        symbol: row.querySelector('.p-symbol')?.value?.trim() || undefined,
        domain: domain
      };
    }).filter(p => p.name);

    const derivedParams = Array.from(section.querySelectorAll('#derivedParamsRows .builder-row')).map(row => ({
      name: row.querySelector('.dp-name')?.value?.trim() || '',
      unit: row.querySelector('.dp-unit')?.value?.trim() || undefined,
      equation: {
        rhs: row.querySelector('.dp-expr')?.value?.trim() || undefined
      }
    })).filter(p => p.name);

    const stateVars = Array.from(section.querySelectorAll('#stateEqRows .builder-row')).map(row => ({
      name: row.querySelector('.sv-name')?.value?.trim() || '',
      symbol: row.querySelector('.sv-symbol')?.value?.trim() || undefined,
      unit: row.querySelector('.sv-unit')?.value?.trim() || undefined,
      initial_value: parseMaybeNumber(row.querySelector('.sv-initial')?.value?.trim()),
      variable_of_interest: row.querySelector('.sv-voi')?.checked || false,
      coupling_variable: row.querySelector('.sv-coupling')?.checked || false,
      equation: {
        rhs: row.querySelector('.sv-expr')?.value?.trim() || undefined
      }
    })).filter(e => e.name);

    const derivedVars = Array.from(section.querySelectorAll('#derivedVarsRows .builder-row')).map(row => ({
      name: row.querySelector('.dv-name')?.value?.trim() || '',
      unit: row.querySelector('.dv-unit')?.value?.trim() || undefined,
      equation: {
        rhs: row.querySelector('.dv-expr')?.value?.trim() || undefined
      }
    })).filter(e => e.name);

    const outputTransforms = Array.from(section.querySelectorAll('#outputTransformsRows .builder-row')).map(row => ({
      name: row.querySelector('.ot-name')?.value?.trim() || '',
      unit: row.querySelector('.ot-unit')?.value?.trim() || undefined,
      equation: {
        rhs: row.querySelector('.ot-expr')?.value?.trim() || undefined
      }
    })).filter(e => e.name);

    const functions = Array.from(section.querySelectorAll('#functionsRows .builder-row')).map(row => ({
      name: row.querySelector('.fn-name')?.value?.trim() || '',
      equation: {
        rhs: row.querySelector('.fn-expr')?.value?.trim() || undefined
      }
    })).filter(f => f.name);

    const couplingTerms = Array.from(section.querySelectorAll('#couplingTermsRows .builder-row')).map(row => ({
      name: row.querySelector('.ct-name')?.value?.trim() || '',
      value: parseMaybeNumber(row.querySelector('.ct-value')?.value?.trim())
    })).filter(c => c.name);

    return {
      model: prune({
        name: name,
        label: name,
        description: description || undefined,
        system_type: systemType,
        parameters: modelParams.length ? modelParams : undefined,
        derived_parameters: derivedParams.length ? derivedParams : undefined,
        state_variables: stateVars.length ? stateVars : undefined,
        derived_variables: derivedVars.length ? derivedVars : undefined,
        output: outputTransforms.length ? outputTransforms : undefined,
        functions: functions.length ? functions : undefined,
        coupling_terms: couplingTerms.length ? couplingTerms : undefined,
      })
    };
  }

  function copyPythonCode() {
    const section = document.getElementById('builderContent');
    if (!section) {
      alert('Please configure a model first');
      return;
    }

    const spec = collectSpec(section, { models: STATE.data || [] });
    const pythonCode = generatePythonCode(spec);

    navigator.clipboard.writeText(pythonCode).then(() => {
      alert('Python code copied to clipboard!');
    }).catch(err => {
      alert('Failed to copy: ' + err.message);
    });
  }

  function downloadYaml() {
    const section = document.getElementById('builderContent');
    if (!section) {
      alert('Please configure a model first');
      return;
    }

    const spec = collectSpec(section, { models: STATE.data || [] });
    const yamlContent = generateYamlContent(spec);
    const modelName = spec.model.name || 'CustomModel';

    const blob = new Blob([yamlContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function generatePythonCode(spec) {
    const model = spec.model;
    const name = model.name || 'CustomModel';

    let code = `# Generated Neural Mass Model: ${name}\n`;
    code += `from tvbo import Dynamics\n\n`;
    code += `# Define model\n`;
    code += `model_dict = {\n`;
    code += `    'name': '${name}',\n`;
    if (model.description) {
      code += `    'description': '${model.description}',\n`;
    }
    if (model.system_type && model.system_type !== 'continuous') {
      code += `    'system_type': '${model.system_type}',\n`;
    }

    if (model.parameters && model.parameters.length > 0) {
      code += `    'parameters': [\n`;
      model.parameters.forEach(p => {
        code += `        {\n`;
        code += `            'name': '${p.name}',\n`;
        code += `            'value': ${p.value},\n`;
        if (p.unit) code += `            'unit': '${p.unit}',\n`;
        if (p.symbol) code += `            'symbol': '${p.symbol}',\n`;
        if (p.domain) {
          code += `            'domain': {\n`;
          if (p.domain.lo !== undefined) code += `                'lo': ${p.domain.lo},\n`;
          if (p.domain.hi !== undefined) code += `                'hi': ${p.domain.hi},\n`;
          code += `            },\n`;
        }
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    if (model.derived_parameters && model.derived_parameters.length > 0) {
      code += `    'derived_parameters': [\n`;
      model.derived_parameters.forEach(dp => {
        code += `        {\n`;
        code += `            'name': '${dp.name}',\n`;
        if (dp.unit) code += `            'unit': '${dp.unit}',\n`;
        if (dp.equation && dp.equation.rhs) {
          code += `            'equation': {'rhs': '${dp.equation.rhs}'},\n`;
        }
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    if (model.state_variables && model.state_variables.length > 0) {
      code += `    'state_variables': [\n`;
      model.state_variables.forEach(sv => {
        code += `        {\n`;
        code += `            'name': '${sv.name}',\n`;
        if (sv.symbol) code += `            'symbol': '${sv.symbol}',\n`;
        if (sv.unit) code += `            'unit': '${sv.unit}',\n`;
        if (sv.initial_value !== undefined) code += `            'initial_value': ${sv.initial_value},\n`;
        if (sv.variable_of_interest !== undefined) code += `            'variable_of_interest': ${sv.variable_of_interest},\n`;
        if (sv.coupling_variable) code += `            'coupling_variable': ${sv.coupling_variable},\n`;
        if (sv.equation && sv.equation.rhs) {
          code += `            'equation': {'rhs': '${sv.equation.rhs}'},\n`;
        }
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    if (model.derived_variables && model.derived_variables.length > 0) {
      code += `    'derived_variables': [\n`;
      model.derived_variables.forEach(dv => {
        code += `        {\n`;
        code += `            'name': '${dv.name}',\n`;
        if (dv.unit) code += `            'unit': '${dv.unit}',\n`;
        if (dv.equation && dv.equation.rhs) {
          code += `            'equation': {'rhs': '${dv.equation.rhs}'},\n`;
        }
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    if (model.output && model.output.length > 0) {
      code += `    'output': [\n`;
      model.output.forEach(ot => {
        code += `        {\n`;
        code += `            'name': '${ot.name}',\n`;
        if (ot.unit) code += `            'unit': '${ot.unit}',\n`;
        if (ot.equation && ot.equation.rhs) {
          code += `            'equation': {'rhs': '${ot.equation.rhs}'},\n`;
        }
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    if (model.functions && model.functions.length > 0) {
      code += `    'functions': [\n`;
      model.functions.forEach(fn => {
        code += `        {\n`;
        code += `            'name': '${fn.name}',\n`;
        if (fn.equation && fn.equation.rhs) {
          code += `            'equation': {'rhs': '${fn.equation.rhs}'},\n`;
        }
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    if (model.coupling_terms && model.coupling_terms.length > 0) {
      code += `    'coupling_terms': [\n`;
      model.coupling_terms.forEach(ct => {
        code += `        {\n`;
        code += `            'name': '${ct.name}',\n`;
        code += `            'value': ${ct.value},\n`;
        code += `        },\n`;
      });
      code += `    ],\n`;
    }

    code += `}\n\n`;
    code += `# Create Dynamics instance\n`;
    code += `model = Dynamics(**model_dict)\n\n`;
    code += `# Run simulation\n`;
    code += `results = model.run(duration=1000)\n`;
    code += `results.plot()\n`;
    return code;
  }

  function generateYamlContent(spec) {
    const model = spec.model;
    let yaml = `# Neural Mass Model: ${model.name}\n`;
    yaml += `name: ${model.name}\n`;
    yaml += `label: ${model.label || model.name}\n`;
    if (model.description) {
      yaml += `description: "${model.description}"\n`;
    }
    if (model.system_type && model.system_type !== 'continuous') {
      yaml += `system_type: ${model.system_type}\n`;
    }

    if (model.parameters && model.parameters.length > 0) {
      yaml += `\nparameters:\n`;
      model.parameters.forEach(p => {
        yaml += `  - name: ${p.name}\n`;
        yaml += `    value: ${p.value}\n`;
        if (p.unit) yaml += `    unit: ${p.unit}\n`;
        if (p.symbol) yaml += `    symbol: ${p.symbol}\n`;
        if (p.domain) {
          yaml += `    domain:\n`;
          if (p.domain.lo !== undefined) yaml += `      lo: ${p.domain.lo}\n`;
          if (p.domain.hi !== undefined) yaml += `      hi: ${p.domain.hi}\n`;
        }
      });
    }

    if (model.derived_parameters && model.derived_parameters.length > 0) {
      yaml += `\nderived_parameters:\n`;
      model.derived_parameters.forEach(dp => {
        yaml += `  - name: ${dp.name}\n`;
        if (dp.unit) yaml += `    unit: ${dp.unit}\n`;
        if (dp.equation && dp.equation.rhs) {
          yaml += `    equation:\n`;
          yaml += `      rhs: "${dp.equation.rhs}"\n`;
        }
      });
    }

    if (model.state_variables && model.state_variables.length > 0) {
      yaml += `\nstate_variables:\n`;
      model.state_variables.forEach(sv => {
        yaml += `  - name: ${sv.name}\n`;
        if (sv.symbol) yaml += `    symbol: ${sv.symbol}\n`;
        if (sv.unit) yaml += `    unit: ${sv.unit}\n`;
        if (sv.initial_value !== undefined) yaml += `    initial_value: ${sv.initial_value}\n`;
        if (sv.variable_of_interest !== undefined) yaml += `    variable_of_interest: ${sv.variable_of_interest}\n`;
        if (sv.coupling_variable) yaml += `    coupling_variable: ${sv.coupling_variable}\n`;
        if (sv.equation && sv.equation.rhs) {
          yaml += `    equation:\n`;
          yaml += `      rhs: "${sv.equation.rhs}"\n`;
        }
      });
    }

    if (model.derived_variables && model.derived_variables.length > 0) {
      yaml += `\nderived_variables:\n`;
      model.derived_variables.forEach(dv => {
        yaml += `  - name: ${dv.name}\n`;
        if (dv.unit) yaml += `    unit: ${dv.unit}\n`;
        if (dv.equation && dv.equation.rhs) {
          yaml += `    equation:\n`;
          yaml += `      rhs: "${dv.equation.rhs}"\n`;
        }
      });
    }

    if (model.output && model.output.length > 0) {
      yaml += `\noutput:\n`;
      model.output.forEach(ot => {
        yaml += `  - name: ${ot.name}\n`;
        if (ot.unit) yaml += `    unit: ${ot.unit}\n`;
        if (ot.equation && ot.equation.rhs) {
          yaml += `    equation:\n`;
          yaml += `      rhs: "${ot.equation.rhs}"\n`;
        }
      });
    }

    if (model.functions && model.functions.length > 0) {
      yaml += `\nfunctions:\n`;
      model.functions.forEach(fn => {
        yaml += `  - name: ${fn.name}\n`;
        if (fn.equation && fn.equation.rhs) {
          yaml += `    equation:\n`;
          yaml += `      rhs: "${fn.equation.rhs}"\n`;
        }
      });
    }

    if (model.coupling_terms && model.coupling_terms.length > 0) {
      yaml += `\ncoupling_terms:\n`;
      model.coupling_terms.forEach(ct => {
        yaml += `  - name: ${ct.name}\n`;
        yaml += `    value: ${ct.value}\n`;
      });
    }

    return yaml;
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

  function toTex(lhs, rhs) {
    let L = lhs || '';
    let R = rhs || '';
    if (!L && !R) return '';
    R = R.replace(/\*\*/g, '^');
    R = R.replace(/\bexp\s*\(([^()]*)\)/g, (m, a) => `e^{${a}}`);
    R = R.replace(/\bsqrt\s*\(([^()]*)\)/g, (m, a) => `\\sqrt{${a}}`);
    R = R.replace(/\b(sin|cos|tan|tanh|log)\b/g, (m, fn) => `\\${fn}`);
    try {
      R = R.replace(/(?<=\b[\w)\}])\*(?=[\w(\{\\])/g, ' \\cdot ');
    } catch {
      R = R.replace(/([0-9A-Za-z_\)\}])\*([0-9A-Za-z_\\\(\{])/g, '$1 \\cdot $2');
    }
    return L && R ? `${L} = ${R}` : (L || R);
  }

  function parseMaybeNumber(v) {
    if (v === '') return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : v;
  }

  function valueToStr(v) {
    if (v === undefined || v === null) return '';
    return typeof v === 'number' ? String(v) : v;
  }

  function escapeHtml(s) {
    return String(s || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function escapeAttr(s) {
    return String(s || '').replace(/"/g, '&quot;');
  }

  // Expose functions globally
  window.initializeBuilder = initializeBuilder;
  window.collectSpec = collectSpec;
  window.copyPythonCode = copyPythonCode;
  window.downloadYaml = downloadYaml;

  // Initialize tab content renderers
  function initializeIntegratorTab() {
    const content = document.getElementById('integratorContent');
    if (!content) return;

    const integrators = window.integratorsData || [];
    content.innerHTML = `
      <div class="builder-field">
        <label>Select Integrator</label>
        <select id="integratorSelect" class="builder-select">
          <option value="">— Create new or select existing —</option>
          ${integrators.map(i => `<option value="${i.id}">${escapeHtml(i.name || i.method)}</option>`).join('')}
        </select>
      </div>
      <div class="builder-field">
        <label>Method</label>
        <input id="integratorMethod" class="builder-input" placeholder="HeunDeterministic" />
      </div>
      <div class="builder-field">
        <label>Step Size</label>
        <input id="integratorStepSize" type="number" step="0.001" class="builder-input" placeholder="0.0122" />
      </div>
      <div class="builder-field">
        <label>Duration (ms)</label>
        <input id="integratorDuration" type="number" class="builder-input" placeholder="1000" />
      </div>
    `;
  }

  function initializeCouplingTab() {
    const content = document.getElementById('couplingContent');
    if (!content) return;

    const couplings = window.couplingsData || [];
    content.innerHTML = `
      <div class="builder-field">
        <label>Select Coupling Function</label>
        <select id="couplingSelect" class="builder-select">
          <option value="">— Create new or select existing —</option>
          ${couplings.map(c => `<option value="${c.id}">${escapeHtml(c.label || c.name)}</option>`).join('')}
        </select>
      </div>
      <div class="builder-field">
        <label>Coupling Name</label>
        <input id="couplingName" class="builder-input" placeholder="Linear" />
      </div>
      <div class="builder-field">
        <label>Coupling Strength</label>
        <input id="couplingStrength" type="number" step="0.01" class="builder-input" placeholder="1.0" />
      </div>
    `;
  }

  function initializeMonitorsTab() {
    const content = document.getElementById('monitorsContent');
    if (!content) return;

    const monitors = window.monitorsData || [];
    content.innerHTML = `
      <div class="builder-field">
        <label>Available Monitors</label>
        <div id="monitorsList">
          ${monitors.map(m => `
            <div class="form-check">
              <input class="form-check-input" type="checkbox" id="monitor_${m.id}" value="${m.id}">
              <label class="form-check-label" for="monitor_${m.id}">
                ${escapeHtml(m.label || m.name)} (period: ${m.period || 'default'})
              </label>
            </div>
          `).join('')}
        </div>
      </div>
      <div class="builder-field">
        <h5>Add New Monitor</h5>
        <label>Monitor Name</label>
        <input id="newMonitorName" class="builder-input" placeholder="Raw" />
      </div>
      <div class="builder-field">
        <label>Sampling Period</label>
        <input id="newMonitorPeriod" type="number" step="0.1" class="builder-input" placeholder="0.9765625" />
      </div>
      <div class="builder-actions">
        <button class="btn btn-sm btn-secondary" id="addMonitor">Add Monitor</button>
      </div>
    `;
  }

  function initializeObservationModelsTab() {
    const content = document.getElementById('observationModelsList');
    if (!content) return;

    const monitors = window.monitorsData || [];

    // Container for active observation models in the pipeline
    content.innerHTML = `
      <div id="observationPipeline" class="observation-pipeline" style="
        display: flex;
        flex-direction: row;
        gap: 15px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 8px;
        min-height: 120px;
        overflow-x: auto;
        align-items: center;
      ">
        <div class="pipeline-placeholder" style="color: #6c757d; font-style: italic;">
          No observation models added yet. Click "Add Observation Model" below.
        </div>
      </div>
    `;

    // Setup add button handler
    const addButton = document.getElementById('addObservationModel');
    if (addButton) {
      addButton.onclick = function() {
        showObservationModelSelector(monitors);
      };
    }
  }

  // Dynamic form builder using Pydantic schema
  async function fetchModelSchema(modelName) {
    try {
      const response = await fetch('/tvbo/api/schema/model/' + modelName, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'call',
          params: {}
        })
      });
      const data = await response.json();
      return data.result;
    } catch (error) {
      console.error('Error fetching schema:', error);
      return null;
    }
  }

  function createFormField(fieldDef, idPrefix) {
    const fieldId = idPrefix + '_' + fieldDef.name;
    let inputHtml = '';

    if (fieldDef.enum_values) {
      // Enum field - dropdown
      inputHtml = `
        <select id="${fieldId}" class="builder-select" style="width: 100%;" ${fieldDef.required ? 'required' : ''}>
          <option value="">— Select ${fieldDef.name} —</option>
          ${fieldDef.enum_values.map(ev => `<option value="${ev.value}">${ev.label}</option>`).join('')}
        </select>
      `;
    } else if (fieldDef.type === 'boolean') {
      inputHtml = `
        <input type="checkbox" id="${fieldId}" class="builder-checkbox" />
      `;
    } else if (fieldDef.type === 'integer') {
      inputHtml = `
        <input type="number" id="${fieldId}" class="builder-input" step="1"
               placeholder="${fieldDef.default || ''}" style="width: 100%;"
               ${fieldDef.required ? 'required' : ''} />
      `;
    } else if (fieldDef.type === 'float') {
      inputHtml = `
        <input type="number" id="${fieldId}" class="builder-input" step="0.01"
               placeholder="${fieldDef.default || ''}" style="width: 100%;"
               ${fieldDef.required ? 'required' : ''} />
      `;
    } else if (fieldDef.type === 'string') {
      if (fieldDef.description && fieldDef.description.length > 100) {
        // Long description - textarea
        inputHtml = `
          <textarea id="${fieldId}" class="builder-input" rows="2"
                    placeholder="${fieldDef.default || ''}" style="width: 100%;"
                    ${fieldDef.required ? 'required' : ''}></textarea>
        `;
      } else {
        inputHtml = `
          <input type="text" id="${fieldId}" class="builder-input"
                 placeholder="${fieldDef.default || ''}" style="width: 100%;"
                 ${fieldDef.required ? 'required' : ''} />
        `;
      }
    } else if (fieldDef.type === 'object') {
      // Nested object - show as placeholder for now
      inputHtml = `
        <input type="text" id="${fieldId}" class="builder-input"
               placeholder="Configure ${fieldDef.name}..." style="width: 100%;" readonly />
        <small class="text-muted">Complex object (not yet editable)</small>
      `;
    } else if (fieldDef.is_list) {
      // List field - show as placeholder
      inputHtml = `
        <input type="text" id="${fieldId}" class="builder-input"
               placeholder="Add ${fieldDef.name}..." style="width: 100%;" readonly />
        <small class="text-muted">List field (configure after creation)</small>
      `;
    } else {
      // Fallback
      inputHtml = `
        <input type="text" id="${fieldId}" class="builder-input"
               placeholder="${fieldDef.default || ''}" style="width: 100%;" />
      `;
    }

    const requiredMarker = fieldDef.required ? '<span style="color: red;">*</span>' : '';
    const descriptionHtml = fieldDef.description
      ? `<small class="text-muted">${escapeHtml(fieldDef.description)}</small>`
      : '';

    return `
      <div class="builder-field" style="margin-bottom: 12px;">
        <label style="font-size: 0.9em; font-weight: 600;">
          ${fieldDef.name} ${requiredMarker}
        </label>
        ${inputHtml}
        ${descriptionHtml}
      </div>
    `;
  }

  function collectFormData(fieldDefs, idPrefix) {
    const data = {};
    fieldDefs.forEach(fieldDef => {
      const fieldId = idPrefix + '_' + fieldDef.name;
      const element = document.getElementById(fieldId);
      if (!element) return;

      if (fieldDef.type === 'boolean') {
        data[fieldDef.name] = element.checked;
      } else if (fieldDef.type === 'integer') {
        const val = parseInt(element.value);
        if (!isNaN(val)) data[fieldDef.name] = val;
      } else if (fieldDef.type === 'float') {
        const val = parseFloat(element.value);
        if (!isNaN(val)) data[fieldDef.name] = val;
      } else {
        const val = element.value.trim();
        if (val) data[fieldDef.name] = val;
      }
    });
    return data;
  }

  async function showObservationModelSelector(monitors) {
    // Create inline form instead of modal
    const pipeline = document.getElementById('observationPipeline');
    if (!pipeline) return;

    // Check if form already exists
    if (document.getElementById('observationModelForm')) return;

    // Remove placeholder if exists
    const placeholder = pipeline.querySelector('.pipeline-placeholder');
    if (placeholder) placeholder.remove();

    // Fetch Monitor schema dynamically
    const schema = await fetchModelSchema('Monitor');
    if (!schema) {
      alert('Could not load Monitor schema');
      return;
    }

    // Filter fields to show (skip complex nested objects for now)
    const simpleFields = schema.fields.filter(f =>
      !['parameters', 'environment', 'transformation', 'pipeline',
        'data_injections', 'argument_mappings', 'derivatives', 'equation'].includes(f.name)
    );

    // Generate form HTML dynamically
    const formFieldsHtml = simpleFields.map(field => createFormField(field, 'modal')).join('');

    // Create inline form
    const formCard = document.createElement('div');
    formCard.id = 'observationModelForm';
    formCard.style.cssText = `
      background: #fff3cd;
      border: 2px dashed #ffc107;
      border-radius: 8px;
      padding: 20px;
      min-width: 400px;
      max-width: 600px;
    `;

    formCard.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 15px; font-size: 1.1em;">Configure Monitor / Observation Model</div>
      <div style="font-size: 0.85em; margin-bottom: 12px; color: #666;">
        <strong>Schema:</strong> ${schema.name} <br>
        <em>${schema.doc || 'Observation model for monitoring simulation output'}</em>
      </div>

      ${formFieldsHtml}

      <div style="display: flex; gap: 8px; margin-top: 15px;">
        <button class="btn btn-sm btn-primary" id="confirmAddModel">Add Model</button>
        <button class="btn btn-sm btn-secondary" id="cancelAddModel">Cancel</button>
      </div>

      <div style="margin-top: 10px; padding: 10px; background: #e7f3ff; border-left: 3px solid #0066cc; font-size: 0.85em;">
        <strong>Note:</strong> After adding, you can configure the processing pipeline and complex attributes on the model card.
      </div>
    `;

    pipeline.appendChild(formCard);

    // Setup confirm button
    document.getElementById('confirmAddModel').onclick = function() {
      const formData = collectFormData(simpleFields, 'modal');

      // Validate required fields
      const missingRequired = simpleFields
        .filter(f => f.required && !formData[f.name])
        .map(f => f.name);

      if (missingRequired.length > 0) {
        alert(`Please fill in required fields: ${missingRequired.join(', ')}`);
        return;
      }

      // Add observation model with collected data
      addObservationModelToPipeline({
        id: Date.now(),
        ...formData,
        // Initialize schema-aligned structures
        transformation: null,
        pipeline: [],
        dataInjections: [],
        argumentMappings: [],
        derivatives: [],
        parameters: []
      });

      formCard.remove();
    };

    // Setup cancel button
    document.getElementById('cancelAddModel').onclick = function() {
      formCard.remove();
      // Restore placeholder if pipeline is empty
      if (pipeline.children.length === 0) {
        pipeline.innerHTML = '<div class="pipeline-placeholder" style="color: #6c757d; font-style: italic;">No observation models added yet. Click "Add Observation Model" below.</div>';
      }
    };
  }

  function addObservationModelToPipeline(modelData) {
    const pipeline = document.getElementById('observationPipeline');
    if (!pipeline) return;

    // Remove placeholder if exists
    const placeholder = pipeline.querySelector('.pipeline-placeholder');
    if (placeholder) placeholder.remove();

    // Initialize schema-aligned structures if not exist
    if (!modelData.pipeline) modelData.pipeline = [];
    if (!modelData.dataInjections) modelData.dataInjections = [];
    if (!modelData.argumentMappings) modelData.argumentMappings = [];
    if (!modelData.derivatives) modelData.derivatives = [];

    // Create model card
    const modelCard = document.createElement('div');
    modelCard.className = 'observation-model-card';
    modelCard.dataset.modelId = modelData.id;
    modelCard.style.cssText = `
      background: white;
      border: 2px solid #0d6efd;
      border-radius: 8px;
      padding: 15px;
      min-width: 300px;
      max-width: 400px;
      position: relative;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    `;

    // Build pipeline steps HTML
    const pipelineHtml = modelData.pipeline.length > 0
      ? modelData.pipeline.map((step, idx) => `
          <div style="font-size: 0.8em; padding: 4px 6px; background: #f8f9fa; margin-top: 4px; border-radius: 3px; display: flex; justify-content: space-between; align-items: center;">
            <span><strong>${step.order}.</strong> ${escapeHtml(step.operation_type || 'transform')}</span>
            <button class="btn btn-sm" style="padding: 0 4px; font-size: 0.8em;" onclick="removeProcessingStep(${modelData.id}, ${idx})">×</button>
          </div>
        `).join('')
      : '<div style="font-size: 0.8em; color: #6c757d; font-style: italic; margin-top: 4px;">No pipeline steps</div>';

    // Build basic info section
    let infoLines = [];
    if (modelData.label) infoLines.push(`Label: ${escapeHtml(modelData.label)}`);
    if (modelData.acronym) infoLines.push(`Acronym: ${escapeHtml(modelData.acronym)}`);
    if (modelData.description) infoLines.push(`Desc: ${escapeHtml(modelData.description)}`);
    if (modelData.imaging_modality) infoLines.push(`Modality: ${modelData.imaging_modality}`);
    if (modelData.time_scale) infoLines.push(`Time: ${modelData.time_scale}`);

    const infoHtml = infoLines.length > 0
      ? `<div style="font-size: 0.75em; color: #666; margin-top: 6px;">${infoLines.join(' • ')}</div>`
      : '';

    // Handle name field
    const displayName = modelData.name || modelData.instanceName || 'Unnamed';
    const displayType = modelData.monitorLabel || 'Monitor';
    const displayPeriod = modelData.period || modelData.samplingPeriod || 'N/A';

    modelCard.innerHTML = `
      <button class="btn btn-sm btn-danger" style="position: absolute; top: 5px; right: 5px; padding: 2px 6px;"
              onclick="removeObservationModel(${modelData.id})">×</button>

      <div style="font-weight: bold; margin-bottom: 3px; font-size: 1.05em;">${escapeHtml(displayName)}</div>
      <div style="font-size: 0.85em; color: #6c757d; margin-bottom: 2px;">${escapeHtml(displayType)}</div>
      ${infoHtml}

      <div style="font-size: 0.8em; margin-top: 8px;">
        <span style="background: #e9ecef; padding: 2px 6px; border-radius: 3px;">
          Period: ${displayPeriod}
        </span>
      </div>

      <hr style="margin: 10px 0;">

      <div style="font-size: 0.85em; font-weight: bold; margin-bottom: 4px;">Processing Pipeline:</div>
      <div class="processing-steps-list" id="steps-${modelData.id}">
        ${pipelineHtml}
      </div>
      <button class="btn btn-sm btn-outline-primary" style="width: 100%; margin-top: 8px; font-size: 0.85em;"
              onclick="addProcessingStepToModel(${modelData.id})">+ Add Pipeline Step</button>

      <hr style="margin: 10px 0;">

      <div style="font-size: 0.8em; color: #666;">
        <div>Data Injections: ${modelData.dataInjections ? modelData.dataInjections.length : 0}</div>
        <div>Argument Mappings: ${modelData.argumentMappings ? modelData.argumentMappings.length : 0}</div>
        <div>Derivatives: ${modelData.derivatives ? modelData.derivatives.length : 0}</div>
      </div>
    `;

    // Add arrow between models if not first
    if (pipeline.children.length > 0) {
      const arrow = document.createElement('div');
      arrow.innerHTML = '→';
      arrow.style.cssText = 'font-size: 24px; color: #6c757d;';
      pipeline.appendChild(arrow);
    }

    pipeline.appendChild(modelCard);

    // Store model data globally for later access
    if (!window.observationModelsData) {
      window.observationModelsData = {};
    }
    window.observationModelsData[modelData.id] = modelData;
  }

  // Global functions for managing observation models and steps
  window.removeObservationModel = function(modelId) {
    const card = document.querySelector(`.observation-model-card[data-model-id="${modelId}"]`);
    if (card) {
      // Remove arrow before this card if it exists
      const prevSibling = card.previousElementSibling;
      if (prevSibling && prevSibling.innerHTML === '→') {
        prevSibling.remove();
      }
      // Or remove arrow after if this is first card
      const nextSibling = card.nextElementSibling;
      if (nextSibling && nextSibling.innerHTML === '→') {
        nextSibling.remove();
      }
      card.remove();
    }

    // Remove from data store
    if (window.observationModelsData) {
      delete window.observationModelsData[modelId];
    }

    // Restore placeholder if pipeline is empty
    const pipeline = document.getElementById('observationPipeline');
    if (pipeline && pipeline.children.length === 0) {
      pipeline.innerHTML = '<div class="pipeline-placeholder" style="color: #6c757d; font-style: italic;">No observation models added yet. Click "Add Observation Model" below.</div>';
    }
  };

  window.addProcessingStepToModel = function(modelId) {
    const modelData = window.observationModelsData?.[modelId];
    if (!modelData) return;

    // Create inline form for adding processing step
    const stepsList = document.getElementById(`steps-${modelId}`);
    if (!stepsList) return;

    // Check if form already exists
    if (document.getElementById(`step-form-${modelId}`)) return;

    const formHtml = `
      <div id="step-form-${modelId}" style="background: #fff3cd; padding: 8px; margin-top: 8px; border-radius: 4px; border: 1px dashed #ffc107;">
        <div style="font-size: 0.85em; margin-bottom: 6px;">
          <label style="display: block; margin-bottom: 2px;">Operation Type</label>
          <select id="stepOpType-${modelId}" class="builder-select" style="width: 100%; font-size: 0.85em;">
            <option value="subsample">Subsample</option>
            <option value="temporal_average">Temporal Average</option>
            <option value="projection">Projection</option>
            <option value="convolution">Convolution</option>
            <option value="select">Select</option>
            <option value="custom_transform">Custom Transform</option>
          </select>
        </div>
        <div style="font-size: 0.85em; margin-bottom: 6px;">
          <label style="display: block; margin-bottom: 2px;">Function/Transform</label>
          <input id="stepFunction-${modelId}" class="builder-input" placeholder="e.g., downsample" style="width: 100%; font-size: 0.85em;" />
        </div>
        <div style="display: flex; gap: 4px; margin-top: 6px;">
          <button class="btn btn-sm btn-success" style="font-size: 0.8em;" onclick="confirmAddProcessingStep(${modelId})">Add</button>
          <button class="btn btn-sm btn-secondary" style="font-size: 0.8em;" onclick="cancelAddProcessingStep(${modelId})">Cancel</button>
        </div>
      </div>
    `;

    stepsList.insertAdjacentHTML('beforeend', formHtml);
  };

  window.confirmAddProcessingStep = function(modelId) {
    const modelData = window.observationModelsData?.[modelId];
    if (!modelData) return;

    const opType = document.getElementById(`stepOpType-${modelId}`)?.value;
    const func = document.getElementById(`stepFunction-${modelId}`)?.value;

    if (!opType) {
      alert('Please select an operation type');
      return;
    }

    const newStep = {
      order: modelData.pipeline.length + 1,
      operation_type: opType,
      function: func || opType,
      input_mapping: [],
      output_alias: null,
      apply_on_dimension: null,
      ensure_shape: null,
      variables_of_interest: []
    };

    modelData.pipeline.push(newStep);

    // Remove form
    document.getElementById(`step-form-${modelId}`)?.remove();

    // Refresh the model card
    refreshObservationModelCard(modelId);
  };

  window.cancelAddProcessingStep = function(modelId) {
    document.getElementById(`step-form-${modelId}`)?.remove();
  };

  window.removeProcessingStep = function(modelId, stepIndex) {
    const modelData = window.observationModelsData?.[modelId];
    if (!modelData) return;

    modelData.pipeline.splice(stepIndex, 1);

    // Re-order remaining steps
    modelData.pipeline.forEach((step, idx) => {
      step.order = idx + 1;
    });

    refreshObservationModelCard(modelId);
  };

  function refreshObservationModelCard(modelId) {
    const modelData = window.observationModelsData?.[modelId];
    if (!modelData) return;

    const stepsList = document.getElementById(`steps-${modelId}`);
    if (!stepsList) return;

    const stepsHtml = modelData.pipeline.length > 0
      ? modelData.pipeline.map((step, idx) => `
          <div style="font-size: 0.8em; padding: 4px 6px; background: #f8f9fa; margin-top: 4px; border-radius: 3px; display: flex; justify-content: space-between; align-items: center;">
            <span><strong>${step.order}.</strong> ${escapeHtml(step.operation_type || 'transform')} ${step.function ? '(' + escapeHtml(step.function) + ')' : ''}</span>
            <button class="btn btn-sm" style="padding: 0 4px; font-size: 0.8em;" onclick="removeProcessingStep(${modelId}, ${idx})">×</button>
          </div>
        `).join('')
      : '<div style="font-size: 0.8em; color: #6c757d; font-style: italic; margin-top: 4px;">No pipeline steps</div>';

    stepsList.innerHTML = stepsHtml;
  }

  function initializeNetworkTab() {
    // Handle mode switching between Custom and Brain Network
    const customPanel = document.getElementById('customNetworkPanel');
    const brainPanel = document.getElementById('brainNetworkPanel');
    const modeRadios = document.querySelectorAll('input[name="networkMode"]');

    if (!customPanel || !brainPanel) return;

    // Mode switching
    modeRadios.forEach(radio => {
      radio.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
          customPanel.style.display = 'block';
          brainPanel.style.display = 'none';
        } else if (e.target.value === 'brain') {
          customPanel.style.display = 'none';
          brainPanel.style.display = 'block';
        }
      });
    });

    // Populate tractogram dropdown from database
    const tractogramSelect = document.getElementById('brainTractogram');
    if (tractogramSelect && window.tractogramsData) {
      // Clear existing options except the first placeholder
      while (tractogramSelect.options.length > 1) {
        tractogramSelect.remove(1);
      }
      window.tractogramsData.forEach(tractogram => {
        const option = document.createElement('option');
        option.value = tractogram;
        option.textContent = tractogram;
        tractogramSelect.appendChild(option);
      });
      // Add custom option at the end
      const customOption = document.createElement('option');
      customOption.value = 'custom';
      customOption.textContent = 'Custom (upload below)';
      tractogramSelect.appendChild(customOption);
    }

    // Populate parcellation dropdown from database
    const parcellationSelect = document.getElementById('brainParcellation');
    if (parcellationSelect && window.parcellationsData) {
      // Clear existing options except the first placeholder
      while (parcellationSelect.options.length > 1) {
        parcellationSelect.remove(1);
      }
      window.parcellationsData.forEach(parc => {
        const option = document.createElement('option');
        option.value = parc.id;
        option.textContent = parc.label;
        parcellationSelect.appendChild(option);
      });
      // Add custom option at the end
      const customOption = document.createElement('option');
      customOption.value = 'custom';
      customOption.textContent = 'Custom (upload below)';
      parcellationSelect.appendChild(customOption);
    }

    // Initialize Custom Network node/edge builders
    initializeCustomNetworkBuilder();

    // Initialize file upload handlers
    initializeNetworkFileUploads();
  }

  function initializeCustomNetworkBuilder() {
    const nodesContainer = document.getElementById('customNetworkNodes');
    const edgesContainer = document.getElementById('customNetworkEdges');
    const addNodeBtn = document.getElementById('addCustomNode');
    const addEdgeBtn = document.getElementById('addCustomEdge');

    if (!nodesContainer || !edgesContainer) return;

    // Prevent multiple initializations
    if (addNodeBtn?.dataset.initialized === 'true') return;
    if (addNodeBtn) addNodeBtn.dataset.initialized = 'true';
    if (addEdgeBtn) addEdgeBtn.dataset.initialized = 'true';

    // Helper to get current nodes for edge dropdowns
    function getCurrentNodes() {
      const nodes = [];
      nodesContainer.querySelectorAll('.builder-row').forEach(row => {
        const id = row.dataset.nodeId || row.querySelector('.node-id')?.value || '';
        const label = row.querySelector('.node-label')?.value || '';
        nodes.push({ id, label: label || `Node ${id}` });
      });
      return nodes;
    }

    // Update all edge source/target dropdowns when nodes change
    function updateEdgeNodeOptions() {
      const nodes = getCurrentNodes();
      edgesContainer.querySelectorAll('.edge-source-select, .edge-target-select').forEach(select => {
        const currentValue = select.value;
        // Clear all but first option
        while (select.options.length > 1) {
          select.remove(1);
        }
        // Add node options
        nodes.forEach(node => {
          const option = document.createElement('option');
          option.value = node.id;
          option.textContent = `${node.id}: ${node.label}`;
          select.appendChild(option);
        });
        // Restore value if still valid
        select.value = currentValue;
      });
    }

    function createNodeRow(id = 0, label = '', x = '', y = '', z = '', dynamics = '') {
      // Get available models for the dropdown
      const allModels = window.searchData || [
        {name: 'Generic2dOscillator'}, {name: 'JansenRit'}, {name: 'WilsonCowan'}, {name: 'Epileptor'}
      ];

      const row = document.createElement('div');
      row.className = 'builder-row mb-2 p-2 border rounded bg-light d-flex align-items-center gap-2';
      row.dataset.nodeId = id;
      row.innerHTML = `
        <span class="badge bg-secondary py-2 px-3 flex-shrink-0">Node ${id}</span>
        <input type="hidden" class="node-id" value="${id}"/>
        <input class="form-control form-control-sm node-label" style="width: 120px;" value="${escapeAttr(label)}" placeholder="Label"/>
        <select class="form-select form-select-sm node-dynamics" style="width: 150px;">
          <option value="">Use configured model</option>
          ${allModels.map(m => {
            const name = m.name || m.label || m;
            return `<option value="${escapeAttr(name)}" ${dynamics === name ? 'selected' : ''}>${escapeHtml(name)}</option>`;
          }).join('')}
        </select>
        <input type="number" step="0.1" class="form-control form-control-sm node-x" style="width: 70px;" value="${x}" placeholder="X"/>
        <input type="number" step="0.1" class="form-control form-control-sm node-y" style="width: 70px;" value="${y}" placeholder="Y"/>
        <input type="number" step="0.1" class="form-control form-control-sm node-z" style="width: 70px;" value="${z}" placeholder="Z"/>
        <button class="btn btn-sm btn-outline-danger flex-shrink-0" title="Remove">×</button>
      `;
      row.querySelector('button').addEventListener('click', () => {
        row.remove();
        renumberNodes();
        updateEdgeNodeOptions();
        updateGraph3D();
      });
      row.querySelector('.node-label').addEventListener('input', updateEdgeNodeOptions);
      // Update 3D graph when coordinates change
      ['node-x', 'node-y', 'node-z'].forEach(cls => {
        row.querySelector('.' + cls)?.addEventListener('input', updateGraph3D);
      });
      return row;
    }

    // Update 3D visualization
    function updateGraph3D() {
      if (window.NetworkGraph3D && window.NetworkGraph3D.update) {
        window.NetworkGraph3D.update();
      }
    }

    // Renumber nodes after deletion to keep IDs sequential
    function renumberNodes() {
      const rows = nodesContainer.querySelectorAll('.builder-row');
      rows.forEach((row, idx) => {
        row.dataset.nodeId = idx;
        row.querySelector('.node-id').value = idx;
        row.querySelector('.badge').textContent = `Node ${idx}`;
      });
    }

    // Get configured coupling from the Network tab
    function getConfiguredCoupling() {
      const couplingSelect = document.querySelector('#couplingContent select');
      return couplingSelect?.value || 'Linear';
    }

    function createEdgeRow(source = '', target = '', weight = '', delay = '', couplingType = '') {
      const nodes = getCurrentNodes();
      const configuredCoupling = getConfiguredCoupling();
      const selectedCoupling = couplingType || '';

      const allCouplings = window.couplingsData || [
        {name: 'Linear'}, {name: 'Sigmoidal'}, {name: 'Difference'}, {name: 'Kuramoto'}
      ];

      const row = document.createElement('div');
      row.className = 'builder-row mb-2 p-2 border rounded bg-light d-flex align-items-center gap-2';
      row.innerHTML = `
        <select class="form-select form-select-sm edge-source-select" style="width: 150px;">
          <option value="">Source...</option>
          ${nodes.map(n => `<option value="${escapeAttr(n.id)}" ${source == n.id ? 'selected' : ''}>${n.id}: ${escapeHtml(n.label)}</option>`).join('')}
        </select>
        <span class="text-muted">→</span>
        <select class="form-select form-select-sm edge-target-select" style="width: 150px;">
          <option value="">Target...</option>
          ${nodes.map(n => `<option value="${escapeAttr(n.id)}" ${target == n.id ? 'selected' : ''}>${n.id}: ${escapeHtml(n.label)}</option>`).join('')}
        </select>
        <input type="number" step="0.01" class="form-control form-control-sm edge-weight" style="width: 80px;" value="${weight}" placeholder="Weight"/>
        <input type="number" step="0.1" class="form-control form-control-sm edge-delay" style="width: 80px;" value="${delay}" placeholder="Delay"/>
        <select class="form-select form-select-sm edge-coupling flex-grow-1">
          <option value="">Use configured (${escapeHtml(configuredCoupling)})</option>
          ${allCouplings.map(c => {
            const name = c.name || c.label || c;
            return `<option value="${escapeAttr(name)}" ${selectedCoupling === name ? 'selected' : ''}>${escapeHtml(name)}</option>`;
          }).join('')}
        </select>
        <button class="btn btn-sm btn-outline-danger flex-shrink-0" title="Remove">×</button>
      `;
      row.querySelector('button').addEventListener('click', () => row.remove());
      return row;
    }

    // Add initial node if empty
    if (nodesContainer.children.length === 0) {
      nodesContainer.appendChild(createNodeRow(0, '', '', '', '', ''));
    }

    // No default edges - user adds when they have multiple nodes
    if (edgesContainer.children.length === 0) {
      // Empty by default
    }

    addNodeBtn?.addEventListener('click', () => {
      const nodes = nodesContainer.querySelectorAll('.builder-row');
      const nextId = nodes.length;
      nodesContainer.appendChild(createNodeRow(nextId, '', '', '', '', ''));
      updateEdgeNodeOptions();
    });

    addEdgeBtn?.addEventListener('click', () => {
      edgesContainer.appendChild(createEdgeRow('', '', '', '', ''));
    });

    // Random network generator
    const generateRandomBtn = document.getElementById('generateRandomNetwork');
    const randomNodeCountInput = document.getElementById('randomNodeCount');

    // Actual surface vertices extracted from mni152_2009.obj
    // These are EXACT coordinates from the brain mesh file
    const brainSurfaceCoords = [
      [-65.5, -4.5, 16.5], [-63.9, -2.5, -5.5], [-62.0, -5.5, 6.6], [-60.0, -26.4, 48.5],
      [-57.7, -58.0, -22.5], [-55.0, -66.0, -16.7], [-53.6, 9.5, 42.0], [-52.8, -59.5, 20.0],
      [-51.1, -62.5, 7.5], [-49.5, -78.5, 32.0], [-48.5, -45.8, -27.0], [-47.0, 14.9, -40.0],
      [-46.0, -49.5, -44.5], [-45.0, -46.1, -41.0], [-44.0, -49.0, -11.3], [-43.0, -26.9, -28.5],
      [-42.5, 53.5, 19.1], [-41.0, 31.0, -17.5], [-39.5, 16.5, 28.0], [-38.0, -25.5, -11.5],
      [-36.5, -17.2, 41.0], [-35.0, 22.2, -7.5], [-33.1, -6.5, 57.0], [-32.5, 55.6, 7.0],
      [-30.5, -54.5, 44.0], [-29.5, 31.3, 49.0], [-28.8, 29.0, 54.5], [-28.0, 34.5, 49.0],
      [-26.6, -13.5, -23.5], [-24.0, -55.5, -19.5], [-23.3, -46.0, -19.5], [-22.0, -29.3, 77.0],
      [-21.0, -64.4, 70.0], [-18.5, -35.7, 5.5], [-18.0, 2.5, 68.8], [-16.5, -89.9, 29.5],
      [-15.0, 14.5, -18.0], [-14.0, 12.5, 21.5], [-13.0, -35.5, 12.0], [-12.0, 65.0, -12.9],
      [-11.3, -42.0, -29.0], [-10.0, -62.5, -40.1], [-9.5, 46.0, 8.3], [-8.0, 6.9, 12.5],
      [-7.0, -2.5, -15.6], [-5.5, 7.0, 28.7], [-4.5, -52.5, 30.6], [-4.0, 62.0, -5.0],
      [-3.0, 51.0, 11.8], [-2.5, -43.9, 48.5], [-2.0, 63.0, 19.5], [-1.0, -47.7, -6.5],
      [-0.4, 31.5, -19.5], [0.5, -60.0, 6.4], [1.5, 57.0, 34.5], [2.0, 21.2, 6.5],
      [2.5, -47.5, 38.5], [3.0, -17.9, -43.5], [3.4, 63.5, -23.0], [4.5, -89.8, 2.5],
      [5.5, -74.5, -27.6], [6.0, -74.7, -17.0], [7.5, 31.5, 25.1], [8.6, 73.0, 8.5],
      [9.5, -85.0, 44.3], [10.5, -30.5, -2.0], [12.5, -7.2, 75.0], [14.0, -98.0, 20.5],
      [15.5, 66.5, -0.7], [17.0, -33.8, -3.0], [18.2, -45.0, -17.0], [19.5, -62.5, 70.4],
      [20.5, 16.9, -26.0], [22.0, 32.0, -17.9], [23.5, 27.0, 61.5], [25.2, -36.5, -17.0],
      [26.0, -96.8, 19.5], [27.0, -43.0, -56.7], [29.1, -1.5, -37.5], [30.8, -77.0, 30.5],
      [33.0, 65.5, -12.5], [37.3, 31.0, 34.0], [39.3, -12.5, 3.5], [40.2, 0.5, 60.5],
      [41.0, -12.5, -38.6], [42.0, 41.8, 31.0], [43.5, -81.0, -35.9], [45.8, -45.5, -44.5],
      [47.5, -80.6, 12.5], [49.0, 49.0, 5.3], [49.5, -3.5, 55.6], [51.0, -31.5, 28.7],
      [52.3, 10.0, 5.5], [53.0, -22.5, 42.5], [54.5, -64.0, 13.5], [56.0, 37.9, 5.0],
      [58.9, -13.5, -38.0], [61.0, -56.5, 37.9], [62.0, -29.0, 40.5], [71.7, -35.5, -2.5],
    ];

    generateRandomBtn?.addEventListener('click', () => {
      const count = parseInt(randomNodeCountInput?.value) || 5;
      const nodeCount = Math.max(2, Math.min(count, brainSurfaceCoords.length)); // Clamp to available coords

      console.log('[ModelBuilder] Generating uniform network with', nodeCount, 'nodes using farthest-point sampling');

      // Clear existing nodes and edges
      nodesContainer.innerHTML = '';
      edgesContainer.innerHTML = '';

      // Farthest-point sampling for uniform distribution across cortex
      // Start with a random point, then iteratively pick the point farthest from all selected points
      const selectedIndices = [];
      const selectedCoords = [];

      // Helper: Euclidean distance squared
      const distSq = (a, b) => (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2;

      // Start with a random first point
      const firstIdx = Math.floor(Math.random() * brainSurfaceCoords.length);
      selectedIndices.push(firstIdx);
      selectedCoords.push(brainSurfaceCoords[firstIdx]);

      // Iteratively select the point farthest from all already-selected points
      while (selectedCoords.length < nodeCount) {
        let maxMinDist = -1;
        let bestIdx = -1;

        for (let i = 0; i < brainSurfaceCoords.length; i++) {
          if (selectedIndices.includes(i)) continue;

          // Find minimum distance to any selected point
          let minDist = Infinity;
          for (const sel of selectedCoords) {
            const d = distSq(brainSurfaceCoords[i], sel);
            if (d < minDist) minDist = d;
          }

          // Track the point with the maximum minimum distance
          if (minDist > maxMinDist) {
            maxMinDist = minDist;
            bestIdx = i;
          }
        }

        if (bestIdx >= 0) {
          selectedIndices.push(bestIdx);
          selectedCoords.push(brainSurfaceCoords[bestIdx]);
        }
      }

      // Generate nodes at uniformly distributed brain surface positions
      const nodes = [];
      for (let i = 0; i < nodeCount; i++) {
        const [x, y, z] = selectedCoords[i];
        const label = `Region ${i + 1}`;
        nodes.push({ id: i, x, y, z, label });
        nodesContainer.appendChild(createNodeRow(i, label, x.toString(), y.toString(), z.toString(), ''));
      }

      // Generate random edges (small-world-ish: each node connects to ~2-4 others)
      const edgeSet = new Set();
      for (let i = 0; i < nodeCount; i++) {
        const numConnections = 2 + Math.floor(Math.random() * 3); // 2-4 connections
        for (let c = 0; c < numConnections; c++) {
          const target = Math.floor(Math.random() * nodeCount);
          if (target !== i) {
            const edgeKey = i < target ? `${i}-${target}` : `${target}-${i}`;
            if (!edgeSet.has(edgeKey)) {
              edgeSet.add(edgeKey);
              const weight = (0.5 + Math.random() * 0.5).toFixed(2);
              const delay = (1 + Math.random() * 5).toFixed(1);
              edgesContainer.appendChild(createEdgeRow(i.toString(), target.toString(), weight, delay, ''));
            }
          }
        }
      }

      console.log('[ModelBuilder] Generated', nodeCount, 'nodes and', edgeSet.size, 'edges');

      // Update edge dropdowns and 3D visualization
      updateEdgeNodeOptions();
      updateGraph3D();
    });
  }

  function initializeNetworkFileUploads() {
    // YAML upload for custom network
    const yamlUpload = document.getElementById('networkYamlUpload');
    yamlUpload?.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = function(event) {
          // Store the YAML content
          window.customNetworkYaml = event.target.result;
          alert('Network YAML loaded successfully!');
        };
        reader.readAsText(file);
      }
    });

    // Brain network file uploads
    const tractogramUpload = document.getElementById('brainTractogramUpload');
    tractogramUpload?.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
        window.brainTractogramFile = file;
        alert(`Tractogram file loaded: ${file.name}`);
      }
    });

    const parcellationUpload = document.getElementById('brainParcellationUpload');
    parcellationUpload?.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (file) {
        window.brainParcellationFile = file;
        alert(`Parcellation file loaded: ${file.name}`);
      }
    });
  }

  function initializeStimulusTab() {
    const content = document.getElementById('stimulusContent');
    if (!content) return;

    content.innerHTML = `
      <div class="builder-field">
        <label>Stimulus Type</label>
        <select id="stimulusType" class="builder-select">
          <option value="">None</option>
          <option value="PulseTrain">Pulse Train</option>
          <option value="DC">DC (Constant)</option>
          <option value="Sine">Sinusoidal</option>
        </select>
      </div>
      <div class="builder-field">
        <label>Amplitude</label>
        <input id="stimulusAmplitude" type="number" step="0.01" class="builder-input" placeholder="1.0" />
      </div>
      <div class="builder-field">
        <label>Target Regions (comma-separated indices)</label>
        <input id="stimulusRegions" class="builder-input" placeholder="0,1,2" />
      </div>
    `;
  }

  function initializePreviewTab() {
    const previewElement = document.getElementById('previewYaml');
    if (!previewElement) return;

    // Get experiment-level fields from the new card above tabs
    const expName = document.getElementById('experimentName')?.value?.trim();
    const expLabel = document.getElementById('experimentLabel')?.value?.trim();
    const expDescription = document.getElementById('experimentDescription')?.value?.trim();
    const expReferences = document.getElementById('experimentReferences')?.value?.trim();

    // Collect configuration from all tabs
    const config = {
      simulation_experiment: {
        name: expName || 'Simulation Experiment',
        label: expLabel || undefined,
        description: expDescription || undefined,
        references: expReferences || undefined,
        dynamics: collectDynamicsConfig(),
        network: collectNetworkConfig(),
        integration: collectIntegrationConfig(),
        observation_models: collectObservationModelsConfig(),
        stimulus: collectStimulusConfig()
      }
    };

    // Convert to YAML-like format
    const yamlText = generateYamlPreview(config);
    previewElement.textContent = yamlText;
  }

  function collectDynamicsConfig() {
    // Try both possible element IDs for model selection
    const modelSelect = document.getElementById('modelSelect') || document.getElementById('builderModel');
    const modelName = modelSelect?.value;
    const specName = document.getElementById('builderSpecName')?.value?.trim();

    // Start with basic config
    const config = {};
    if (modelName) config.model = modelName;
    if (specName) config.name = specName;

    // Collect parameters from the model builder UI
    // Classes: p-name, p-value, p-unit, p-symbol, p-domain-lo, p-domain-hi
    const paramRows = document.querySelectorAll('#modelParamsRows .builder-row');
    if (paramRows.length > 0) {
      config.parameters = [];
      paramRows.forEach(row => {
        const name = row.querySelector('.p-name')?.value;
        const value = row.querySelector('.p-value')?.value;
        const unit = row.querySelector('.p-unit')?.value;
        const symbol = row.querySelector('.p-symbol')?.value;
        const domainLo = row.querySelector('.p-domain-lo')?.value;
        const domainHi = row.querySelector('.p-domain-hi')?.value;
        if (name) {
          const param = { name: name };
          if (value) param.value = parseFloat(value);
          if (unit) param.unit = unit;
          if (symbol) param.symbol = symbol;
          if (domainLo || domainHi) {
            param.domain = {};
            if (domainLo) param.domain.lo = parseFloat(domainLo);
            if (domainHi) param.domain.hi = parseFloat(domainHi);
          }
          config.parameters.push(param);
        }
      });
    }

    // Collect state variables from the model builder UI
    // Classes: sv-name, sv-expr, sv-symbol, sv-unit, sv-initial, sv-voi, sv-coupling
    const svRows = document.querySelectorAll('#stateEqRows .builder-row');
    if (svRows.length > 0) {
      config.state_variables = [];
      svRows.forEach(row => {
        const name = row.querySelector('.sv-name')?.value;
        const expr = row.querySelector('.sv-expr')?.value;
        const symbol = row.querySelector('.sv-symbol')?.value;
        const unit = row.querySelector('.sv-unit')?.value;
        const initial = row.querySelector('.sv-initial')?.value;
        const voi = row.querySelector('.sv-voi')?.checked;
        const coupling = row.querySelector('.sv-coupling')?.checked;
        if (name) {
          const sv = { name: name };
          if (expr) sv.equation = { rhs: expr };
          if (symbol) sv.symbol = symbol;
          if (unit) sv.unit = unit;
          if (initial) sv.initial_value = parseFloat(initial);
          if (voi !== undefined) sv.variable_of_interest = voi;
          if (coupling !== undefined) sv.coupling_variable = coupling;
          config.state_variables.push(sv);
        }
      });
    }

    // Collect derived parameters
    // Classes: dp-name, dp-expr, dp-unit
    const dpRows = document.querySelectorAll('#derivedParamsRows .builder-row');
    if (dpRows.length > 0) {
      config.derived_parameters = [];
      dpRows.forEach(row => {
        const name = row.querySelector('.dp-name')?.value;
        const expr = row.querySelector('.dp-expr')?.value;
        const unit = row.querySelector('.dp-unit')?.value;
        if (name) {
          const dp = { name: name };
          if (expr) dp.equation = { rhs: expr };
          if (unit) dp.unit = unit;
          config.derived_parameters.push(dp);
        }
      });
    }

    // Collect derived variables
    // Classes: dv-name, dv-expr, dv-unit
    const dvRows = document.querySelectorAll('#derivedVarsRows .builder-row');
    if (dvRows.length > 0) {
      config.derived_variables = [];
      dvRows.forEach(row => {
        const name = row.querySelector('.dv-name')?.value;
        const expr = row.querySelector('.dv-expr')?.value;
        const unit = row.querySelector('.dv-unit')?.value;
        if (name) {
          const dv = { name: name };
          if (expr) dv.equation = { rhs: expr };
          if (unit) dv.unit = unit;
          config.derived_variables.push(dv);
        }
      });
    }

    // Collect functions
    // Classes: fn-name, fn-expr
    const fnRows = document.querySelectorAll('#functionsRows .builder-row');
    if (fnRows.length > 0) {
      config.functions = [];
      fnRows.forEach(row => {
        const name = row.querySelector('.fn-name')?.value;
        const expr = row.querySelector('.fn-expr')?.value;
        if (name) {
          const fn = { name: name };
          if (expr) fn.equation = { rhs: expr };
          config.functions.push(fn);
        }
      });
    }

    return config;
  }

  function collectNetworkConfig() {
    // Check which mode is selected via radio buttons
    const customRadio = document.getElementById('networkModeCustom');
    const brainRadio = document.getElementById('networkModeBrain');

    // Determine mode from radio buttons, fallback to old select for backwards compat
    let networkMode = 'custom';
    if (brainRadio?.checked) {
      networkMode = 'brain';
    } else if (customRadio?.checked) {
      networkMode = 'custom';
    } else {
      // Fallback to old select element
      networkMode = document.getElementById('networkMode')?.value || 'custom';
    }

    if (networkMode === 'brain') {
      const tractogram = document.getElementById('brainTractogram');
      const parcellation = document.getElementById('brainParcellation');
      const numRegions = document.getElementById('brainNumRegions');
      const globalCoupling = document.getElementById('brainGlobalCoupling');
      const conductionSpeed = document.getElementById('brainConductionSpeed');

      return {
        mode: 'brain',
        tractogram: tractogram?.value || undefined,
        parcellation: parcellation?.value || undefined,
        number_of_regions: numRegions?.value ? parseInt(numRegions.value) : undefined,
        global_coupling_strength: globalCoupling?.value ? parseFloat(globalCoupling.value) : undefined,
        conduction_speed: conductionSpeed?.value ? parseFloat(conductionSpeed.value) : undefined
      };
    } else if (networkMode === 'custom') {
      const label = document.getElementById('customNetworkLabel')?.value;
      const globalCoupling = document.getElementById('customGlobalCoupling')?.value;
      const conductionSpeed = document.getElementById('customConductionSpeed')?.value;

      // Collect nodes
      const nodeRows = document.querySelectorAll('#customNetworkNodes .builder-row');
      const nodes = Array.from(nodeRows).map(row => ({
        id: parseInt(row.querySelector('.node-id')?.value) || 0,
        label: row.querySelector('.node-label')?.value || '',
        position: {
          x: parseFloat(row.querySelector('.node-x')?.value) || 0,
          y: parseFloat(row.querySelector('.node-y')?.value) || 0,
          z: parseFloat(row.querySelector('.node-z')?.value) || 0
        },
        dynamics: {
          name: row.querySelector('.node-dynamics')?.value || undefined,
          dataLocation: row.querySelector('.node-dynamics')?.value
            ? `database/models/${row.querySelector('.node-dynamics')?.value}.yaml`
            : undefined
        }
      }));

      // Collect edges
      const edgeRows = document.querySelectorAll('#customNetworkEdges .builder-row');
      const edges = Array.from(edgeRows).map(row => ({
        source: parseInt(row.querySelector('.edge-source-select')?.value) || 0,
        target: parseInt(row.querySelector('.edge-target-select')?.value) || 1,
        weight: parseFloat(row.querySelector('.edge-weight')?.value) || 1.0,
        delay: parseFloat(row.querySelector('.edge-delay')?.value) || 0,
        coupling: {
          name: row.querySelector('.edge-coupling')?.value || 'Linear'
        }
      }));

      // Check if YAML was uploaded
      if (window.customNetworkYaml) {
        return {
          mode: 'yaml',
          yaml_content: window.customNetworkYaml,
          source: 'yaml_file'
        };
      }

      // Only return if nodes defined
      if (nodes.length === 0) {
        return { mode: 'not configured' };
      }

      return {
        mode: 'custom',
        label: label || 'Custom Network',
        nodes: nodes,
        edges: edges,
        number_of_nodes: nodes.length,
        global_coupling_strength: globalCoupling ? parseFloat(globalCoupling) : 1.0,
        conduction_speed: conductionSpeed ? parseFloat(conductionSpeed) : 3.0
      };
    }

    return { mode: 'not configured' };
  }

  function collectIntegrationConfig() {
    const integratorMethod = document.getElementById('integratorMethod');
    const stepSize = document.getElementById('integratorStepSize');
    const duration = document.getElementById('integratorDuration');

    // Only return configured values
    const config = {};
    if (integratorMethod?.value) config.method = integratorMethod.value;
    if (stepSize?.value) config.step_size = parseFloat(stepSize.value);
    if (duration?.value) config.duration = parseFloat(duration.value);

    return config;
  }

  function collectObservationModelsConfig() {
    const models = [];
    if (window.observationModelsData) {
      Object.values(window.observationModelsData).forEach(modelData => {
        const modelConfig = {
          name: modelData.name || modelData.instanceName || 'unnamed',
          monitor_type: modelData.monitorLabel || 'Monitor'
        };

        // Add all present fields dynamically (handle both naming conventions)
        const fieldMap = {
          'label': 'label',
          'acronym': 'acronym',
          'description': 'description',
          'imaging_modality': 'imaging_modality',
          'imagingModality': 'imaging_modality',
          'period': 'period',
          'samplingPeriod': 'period',
          'time_scale': 'time_scale',
          'timeScale': 'time_scale'
        };

        Object.keys(fieldMap).forEach(sourceKey => {
          if (modelData[sourceKey]) {
            modelConfig[fieldMap[sourceKey]] = modelData[sourceKey];
          }
        });

        // Add pipeline steps
        if (modelData.pipeline && modelData.pipeline.length > 0) {
          modelConfig.pipeline = modelData.pipeline.map(step => {
            const stepConfig = {
              order: step.order,
              operation_type: step.operation_type,
              function: step.function
            };
            if (step.output_alias) stepConfig.output_alias = step.output_alias;
            if (step.apply_on_dimension) stepConfig.apply_on_dimension = step.apply_on_dimension;
            if (step.ensure_shape) stepConfig.ensure_shape = step.ensure_shape;
            return stepConfig;
          });
        }

        // Add other ObservationModel attributes
        if (modelData.dataInjections && modelData.dataInjections.length > 0) {
          modelConfig.data_injections = modelData.dataInjections;
        }
        if (modelData.argumentMappings && modelData.argumentMappings.length > 0) {
          modelConfig.argument_mappings = modelData.argumentMappings;
        }
        if (modelData.derivatives && modelData.derivatives.length > 0) {
          modelConfig.derivatives = modelData.derivatives;
        }

        models.push(modelConfig);
      });
    }
    return models.length > 0 ? models : null;
  }

  function collectStimulusConfig() {
    const stimulusType = document.getElementById('stimulusType');
    const amplitude = document.getElementById('stimulusAmplitude');
    const regions = document.getElementById('stimulusRegions');

    // Only return configured values, skip if no stimulus type selected
    if (!stimulusType?.value || stimulusType.value === '' || stimulusType.value === 'none') {
      return null;
    }

    const config = { type: stimulusType.value };
    if (amplitude?.value) config.amplitude = parseFloat(amplitude.value);
    if (regions?.value) config.target_regions = regions.value.split(',').map(r => parseInt(r.trim())).filter(r => !isNaN(r));

    return config;
  }

  function generateYamlPreview(config) {
    const exp = config.simulation_experiment;
    let yaml = 'simulation_experiment:';

    // Basic experiment fields
    if (exp.name) {
      yaml += `\n  name: "${exp.name}"`;
    }
    if (exp.label) {
      yaml += `\n  label: "${exp.label}"`;
    }
    if (exp.description) {
      yaml += `\n  description: "${exp.description}"`;
    }
    if (exp.references) {
      yaml += `\n  references: "${exp.references}"`;
    }

    // Dynamics - only show if configured
    const dynamics = exp.dynamics;
    if (dynamics && (dynamics.model || dynamics.name)) {
      yaml += `\n\n  local_dynamics:`;
      if (dynamics.model) yaml += `\n    name: ${dynamics.model}`;
      else if (dynamics.name) yaml += `\n    name: ${dynamics.name}`;

      // Show parameters if available
      if (dynamics.parameters && dynamics.parameters.length > 0) {
        yaml += `\n    parameters:`;
        dynamics.parameters.forEach(p => {
          yaml += `\n      - name: ${p.name}`;
          if (p.value !== undefined) yaml += `\n        value: ${p.value}`;
          if (p.unit) yaml += `\n        unit: "${p.unit}"`;
          if (p.symbol) yaml += `\n        symbol: "${p.symbol}"`;
          if (p.domain) {
            yaml += `\n        domain:`;
            if (p.domain.lo !== undefined) yaml += `\n          lo: ${p.domain.lo}`;
            if (p.domain.hi !== undefined) yaml += `\n          hi: ${p.domain.hi}`;
          }
        });
      }

      // Show state variables if available
      if (dynamics.state_variables && dynamics.state_variables.length > 0) {
        yaml += `\n    state_variables:`;
        dynamics.state_variables.forEach(sv => {
          yaml += `\n      - name: ${sv.name}`;
          if (sv.equation && sv.equation.rhs) yaml += `\n        equation: "${sv.equation.rhs}"`;
          if (sv.symbol) yaml += `\n        symbol: "${sv.symbol}"`;
          if (sv.unit) yaml += `\n        unit: "${sv.unit}"`;
          if (sv.initial_value !== undefined) yaml += `\n        initial_value: ${sv.initial_value}`;
          if (sv.variable_of_interest !== undefined) yaml += `\n        variable_of_interest: ${sv.variable_of_interest}`;
          if (sv.coupling_variable !== undefined) yaml += `\n        coupling_variable: ${sv.coupling_variable}`;
        });
      }

      // Show derived parameters if available
      if (dynamics.derived_parameters && dynamics.derived_parameters.length > 0) {
        yaml += `\n    derived_parameters:`;
        dynamics.derived_parameters.forEach(dp => {
          yaml += `\n      - name: ${dp.name}`;
          if (dp.equation && dp.equation.rhs) yaml += `\n        equation: "${dp.equation.rhs}"`;
          if (dp.unit) yaml += `\n        unit: "${dp.unit}"`;
        });
      }

      // Show derived variables if available
      if (dynamics.derived_variables && dynamics.derived_variables.length > 0) {
        yaml += `\n    derived_variables:`;
        dynamics.derived_variables.forEach(dv => {
          yaml += `\n      - name: ${dv.name}`;
          if (dv.equation && dv.equation.rhs) yaml += `\n        equation: "${dv.equation.rhs}"`;
          if (dv.unit) yaml += `\n        unit: "${dv.unit}"`;
        });
      }

      // Show functions if available
      if (dynamics.functions && dynamics.functions.length > 0) {
        yaml += `\n    functions:`;
        dynamics.functions.forEach(fn => {
          yaml += `\n      - name: ${fn.name}`;
          if (fn.equation && fn.equation.rhs) yaml += `\n        equation: "${fn.equation.rhs}"`;
        });
      }
    }

    // Network - only show if there's actual configured content
    const networkConfig = exp.network;
    const hasNetworkContent = networkConfig && (
      networkConfig.mode === 'custom' ||
      networkConfig.mode === 'yaml' ||
      networkConfig.mode === 'brain' ||
      (networkConfig.mode === 'standard' && (
        networkConfig.network_id ||
        networkConfig.parcellation ||
        networkConfig.tractogram ||
        networkConfig.number_of_regions ||
        networkConfig.global_coupling_strength ||
        networkConfig.conduction_speed
      ))
    );

    if (hasNetworkContent) {
      yaml += `\n\n  network:`;

      if (networkConfig.mode === 'brain') {
        if (networkConfig.tractogram) yaml += `\n    tractogram: "${networkConfig.tractogram}"`;
        if (networkConfig.parcellation) yaml += `\n    parcellation: "${networkConfig.parcellation}"`;
        if (networkConfig.number_of_regions) yaml += `\n    number_of_regions: ${networkConfig.number_of_regions}`;
        if (networkConfig.global_coupling_strength) {
          yaml += `\n    global_coupling_strength:`;
          yaml += `\n      name: global_coupling_strength`;
          yaml += `\n      value: ${networkConfig.global_coupling_strength}`;
        }
        if (networkConfig.conduction_speed) {
          yaml += `\n    conduction_speed:`;
          yaml += `\n      name: conduction_speed`;
          yaml += `\n      value: ${networkConfig.conduction_speed}`;
        }
      } else if (networkConfig.mode === 'custom') {
        if (networkConfig.label) yaml += `\n    label: "${networkConfig.label}"`;
        if (networkConfig.number_of_nodes) yaml += `\n    number_of_nodes: ${networkConfig.number_of_nodes}`;
        if (networkConfig.global_coupling_strength) {
          yaml += `\n    global_coupling_strength:`;
          yaml += `\n      name: global_coupling_strength`;
          yaml += `\n      value: ${networkConfig.global_coupling_strength}`;
        }
        if (networkConfig.conduction_speed) {
          yaml += `\n    conduction_speed:`;
          yaml += `\n      name: conduction_speed`;
          yaml += `\n      value: ${networkConfig.conduction_speed}`;
        }

        if (networkConfig.nodes && networkConfig.nodes.length > 0) {
          yaml += `\n    nodes:`;
          networkConfig.nodes.forEach(node => {
            yaml += `\n      - id: ${node.id}`;
            if (node.label) yaml += `\n        label: "${node.label}"`;
            if (node.region) yaml += `\n        region: "${node.region}"`;
            if (node.position) {
              yaml += `\n        position:`;
              yaml += `\n          x: ${node.position.x}`;
              yaml += `\n          y: ${node.position.y}`;
              yaml += `\n          z: ${node.position.z}`;
            }
            if (node.dynamics && node.dynamics.name) {
              yaml += `\n        dynamics:`;
              yaml += `\n          name: ${node.dynamics.name}`;
              if (node.dynamics.dataLocation) {
                yaml += `\n          dataLocation: ${node.dynamics.dataLocation}`;
              }
            }
          });
        }

        if (networkConfig.edges && networkConfig.edges.length > 0) {
          yaml += `\n    edges:`;
          networkConfig.edges.forEach(edge => {
            yaml += `\n      - source: ${edge.source}`;
            yaml += `\n        target: ${edge.target}`;
            if (edge.weight) yaml += `\n        weight: ${edge.weight}`;
            if (edge.delay) yaml += `\n        delay: ${edge.delay}`;
            if (edge.coupling && edge.coupling.name) {
              yaml += `\n        coupling:`;
              yaml += `\n          name: ${edge.coupling.name}`;
            }
          });
        }
      } else if (networkConfig.mode === 'yaml') {
        yaml += `\n    # Network loaded from YAML file`;
        yaml += `\n    source: yaml_file`;
      } else if (networkConfig.mode === 'standard') {
        if (networkConfig.network_id) yaml += `\n    name: ${networkConfig.network_id}`;
        if (networkConfig.parcellation) yaml += `\n    parcellation: "${networkConfig.parcellation}"`;
        if (networkConfig.tractogram) yaml += `\n    tractogram: "${networkConfig.tractogram}"`;
        if (networkConfig.number_of_regions) yaml += `\n    number_of_regions: ${networkConfig.number_of_regions}`;
        if (networkConfig.global_coupling_strength) {
          yaml += `\n    global_coupling_strength:`;
          yaml += `\n      name: global_coupling_strength`;
          yaml += `\n      value: ${networkConfig.global_coupling_strength}`;
        }
        if (networkConfig.conduction_speed) {
          yaml += `\n    conduction_speed:`;
          yaml += `\n      name: conduction_speed`;
          yaml += `\n      value: ${networkConfig.conduction_speed}`;
        }
        if (networkConfig.normalization) yaml += `\n    normalization: "${networkConfig.normalization}"`;
      }
    }

    // Integration - always include with defaults if not configured
    const integration = exp.integration || {};
    yaml += `\n\n  integration:`;
    yaml += `\n    method: ${integration.method || 'Heun'}`;
    yaml += `\n    step_size: ${integration.step_size || 0.1}`;
    yaml += `\n    duration: ${integration.duration || 100}`;

    // Monitors - only show if configured
    const monitors = exp.observation_models;
    if (monitors && Array.isArray(monitors) && monitors.length > 0) {
      yaml += `\n\n  monitors:`;
      monitors.forEach(m => {
        yaml += `\n    - name: ${m.name}`;
        if (m.monitor_type) yaml += `\n      monitor_type: ${m.monitor_type}`;
        if (m.period) yaml += `\n      period: ${m.period}`;
        if (m.label) yaml += `\n      label: "${m.label}"`;
        if (m.acronym) yaml += `\n      acronym: "${m.acronym}"`;
        if (m.description) yaml += `\n      description: "${m.description}"`;
        if (m.imaging_modality) yaml += `\n      imaging_modality: "${m.imaging_modality}"`;

        if (m.pipeline && m.pipeline.length > 0) {
          yaml += `\n      pipeline:`;
          m.pipeline.forEach(step => {
            yaml += `\n        - order: ${step.order}`;
            yaml += `\n          operation_type: ${step.operation_type}`;
            if (step.function) yaml += `\n          function: ${step.function}`;
            if (step.output_alias) yaml += `\n          output_alias: "${step.output_alias}"`;
            if (step.apply_on_dimension) yaml += `\n          apply_on_dimension: ${step.apply_on_dimension}`;
            if (step.ensure_shape) yaml += `\n          ensure_shape: ${step.ensure_shape}`;
          });
        }
      });
    }

    // Stimulus - only show if configured
    const stimulus = exp.stimulus;
    if (stimulus && stimulus.type && stimulus.type !== 'none') {
      yaml += `\n\n  stimulation:`;
      yaml += `\n    type: ${stimulus.type}`;
      if (stimulus.amplitude) yaml += `\n    amplitude: ${stimulus.amplitude}`;
      if (stimulus.target_regions && stimulus.target_regions.length > 0) {
        yaml += `\n    target_regions: [${stimulus.target_regions.join(', ')}]`;
      }
    }

    return yaml;
  }

  // ============================================
  // SIMULATION RUNNER
  // ============================================

  let simulationResults = null;

  /**
   * Collect the full experiment configuration for running a simulation.
   *
   * REQUIRED fields (must be configured):
   *   - local_dynamics.name: Model name (e.g., 'Generic2dOscillator', 'JansenRit')
   *     Source: Model tab -> Base Model dropdown OR custom model name
   *
   *   - network.nodes: At least 2 nodes with positions
   *     Source: Network tab -> Custom network nodes
   *     Each node needs: id, label, position {x, y, z}
   *
   *   - network.edges: Connections between nodes
   *     Source: Network tab -> Edges section
   *     Each edge needs: source, target, weight
   *
   * OPTIONAL fields (have sensible defaults):
   *   - local_dynamics.parameters: Model parameters
   *     Default: Uses model's default parameter values
   *
   *   - integration.method: Integrator type
   *     Default: 'Heun'
   *     Options: Euler, Heun, Dopri5, Dopri853
   *
   *   - integration.step_size: Time step in ms
   *     Default: 0.1 (can be overridden in Run tab)
   *
   *   - integration.duration: Simulation duration in ms
   *     Default: 1000.0 (can be overridden in Run tab)
   *
   *   - coupling.name: Coupling function
   *     Default: 'Linear'
   *
   *   - coupling.global_coupling: Global coupling strength
   *     Default: 1.0
   *
   *   - network.conduction_speed: Signal propagation speed (m/s)
   *     Default: 3.0
   *
   *   - monitors: Observation models / monitors
   *     Default: Raw monitor with period=1.0
   *
   *   - stimulus: External stimulation
   *     Default: None
   *
   * @returns {Object} Experiment configuration for TVBO API
   */
  function collectFullExperiment() {
    console.log('[ModelBuilder] collectFullExperiment called');

    // 1. Collect local dynamics (REQUIRED)
    const dynamicsConfig = collectDynamicsConfig();
    console.log('[ModelBuilder] dynamicsConfig:', dynamicsConfig);

    if (!dynamicsConfig.model && !dynamicsConfig.name) {
      throw new Error('No model selected. Please select or configure a model.');
    }

    const local_dynamics = {
      name: dynamicsConfig.model || dynamicsConfig.name,
    };

    // Collect parameters from the model builder UI
    const paramRows = document.querySelectorAll('#modelParamsRows .builder-row');
    if (paramRows.length > 0) {
      local_dynamics.parameters = [];
      paramRows.forEach(row => {
        const name = row.querySelector('.p-name')?.value;
        const value = row.querySelector('.p-value')?.value;
        const unit = row.querySelector('.p-unit')?.value;
        const symbol = row.querySelector('.p-symbol')?.value;
        const domainLo = row.querySelector('.p-domain-lo')?.value;
        const domainHi = row.querySelector('.p-domain-hi')?.value;
        if (name) {
          const param = { name: name };
          if (value) param.value = parseFloat(value);
          if (unit) param.unit = unit;
          if (symbol) param.symbol = symbol;
          if (domainLo || domainHi) {
            param.domain = {};
            if (domainLo) param.domain.lo = parseFloat(domainLo);
            if (domainHi) param.domain.hi = parseFloat(domainHi);
          }
          local_dynamics.parameters.push(param);
        }
      });
    }

    // Collect state variables from the model builder UI
    // Classes: sv-name, sv-expr, sv-symbol, sv-unit, sv-initial, sv-voi, sv-coupling
    const svRows = document.querySelectorAll('#stateEqRows .builder-row');
    if (svRows.length > 0) {
      local_dynamics.state_variables = [];
      svRows.forEach(row => {
        const name = row.querySelector('.sv-name')?.value;
        const expr = row.querySelector('.sv-expr')?.value;
        const symbol = row.querySelector('.sv-symbol')?.value;
        const unit = row.querySelector('.sv-unit')?.value;
        const initial = row.querySelector('.sv-initial')?.value;
        const voi = row.querySelector('.sv-voi')?.checked;
        const coupling = row.querySelector('.sv-coupling')?.checked;
        if (name) {
          const sv = { name: name };
          if (expr) sv.equation = { rhs: expr };
          if (symbol) sv.symbol = symbol;
          if (unit) sv.unit = unit;
          if (initial) sv.initial_value = parseFloat(initial);
          if (voi !== undefined) sv.variable_of_interest = voi;
          if (coupling !== undefined) sv.coupling_variable = coupling;
          local_dynamics.state_variables.push(sv);
        }
      });
    }

    // Collect derived parameters
    // Classes: dp-name, dp-expr, dp-unit
    const dpRows = document.querySelectorAll('#derivedParamsRows .builder-row');
    if (dpRows.length > 0) {
      local_dynamics.derived_parameters = [];
      dpRows.forEach(row => {
        const name = row.querySelector('.dp-name')?.value;
        const expr = row.querySelector('.dp-expr')?.value;
        const unit = row.querySelector('.dp-unit')?.value;
        if (name) {
          const dp = { name: name };
          if (expr) dp.equation = { rhs: expr };
          if (unit) dp.unit = unit;
          local_dynamics.derived_parameters.push(dp);
        }
      });
    }

    // Collect derived variables
    // Classes: dv-name, dv-expr, dv-unit
    const dvRows = document.querySelectorAll('#derivedVarsRows .builder-row');
    if (dvRows.length > 0) {
      local_dynamics.derived_variables = [];
      dvRows.forEach(row => {
        const name = row.querySelector('.dv-name')?.value;
        const expr = row.querySelector('.dv-expr')?.value;
        const unit = row.querySelector('.dv-unit')?.value;
        if (name) {
          const dv = { name: name };
          if (expr) dv.equation = { rhs: expr };
          if (unit) dv.unit = unit;
          local_dynamics.derived_variables.push(dv);
        }
      });
    }

    // Collect functions
    // Classes: fn-name, fn-expr
    const fnRows = document.querySelectorAll('#functionsRows .builder-row');
    if (fnRows.length > 0) {
      local_dynamics.functions = [];
      fnRows.forEach(row => {
        const name = row.querySelector('.fn-name')?.value;
        const expr = row.querySelector('.fn-expr')?.value;
        if (name) {
          const fn = { name: name };
          if (expr) fn.equation = { rhs: expr };
          local_dynamics.functions.push(fn);
        }
      });
    }

    // 2. Collect network (REQUIRED)
    const networkConfig = collectNetworkConfig();
    console.log('[ModelBuilder] networkConfig:', networkConfig);

    const modelName = local_dynamics.name;
    const couplingName = document.getElementById('couplingFunction')?.value;

    if (!networkConfig || networkConfig.mode === 'not configured') {
      throw new Error('No network configured. Please configure a network in the Network tab.');
    }

    // Build network using matrix format (compatible with API)
    const nNodes = networkConfig.number_of_nodes || networkConfig.nodes?.length || 2;

    // Initialize weights and lengths matrices as flat arrays (row-major order)
    const weights = new Array(nNodes * nNodes).fill(0.0);
    const lengths = new Array(nNodes * nNodes).fill(0.0);

    // Populate from edges
    if (networkConfig.edges && networkConfig.edges.length > 0) {
      networkConfig.edges.forEach(e => {
        const src = parseInt(e.source);
        const tgt = parseInt(e.target);
        if (src >= 0 && src < nNodes && tgt >= 0 && tgt < nNodes) {
          const idx = src * nNodes + tgt;
          weights[idx] = e.weight ?? 1.0;
          // Use delay as length proxy, or compute from positions if available
          lengths[idx] = e.delay ?? 10.0;
        }
      });
    }

    // Build region labels from nodes if available
    const regionLabels = [];
    if (networkConfig.nodes && networkConfig.nodes.length > 0) {
      networkConfig.nodes.forEach((n, idx) => {
        regionLabels.push(n.label || `Region_${idx}`);
      });
    }

    const network = {
      label: networkConfig.label,
      number_of_regions: nNodes,
      weights: { values: weights },
      lengths: { values: lengths },
      node_labels: regionLabels.length > 0 ? regionLabels : undefined,
    };

    // Add global parameters if specified
    if (networkConfig.global_coupling_strength) {
      network.global_coupling_strength = { name: 'global_coupling_strength', value: networkConfig.global_coupling_strength };
    }
    if (networkConfig.conduction_speed) {
      network.conduction_speed = { name: 'conduction_speed', value: networkConfig.conduction_speed };
    }

    console.log('[ModelBuilder] Built network:', network);

    // 3. Collect integration (REQUIRED) - always include with defaults
    const integrationConfig = collectIntegrationConfig();
    console.log('[ModelBuilder] integrationConfig:', integrationConfig);

    const integration = {
      method: integrationConfig.method || 'Heun',
      step_size: integrationConfig.step_size || 0.1,
      duration: integrationConfig.duration || 100,
    };

    // 4. Collect coupling
    const couplingSelect = document.getElementById('couplingFunction');
    const coupling = {
      name: couplingSelect?.value || 'Linear',
    };
    console.log('[ModelBuilder] coupling:', coupling);

    // 5. Collect monitors (optional)
    const monitors = collectObservationModelsConfig();
    console.log('[ModelBuilder] monitors:', monitors);

    // 6. Collect stimulus (optional)
    const stimulus = collectStimulusConfig();
    console.log('[ModelBuilder] stimulus:', stimulus);

    // Build the full experiment object
    const experiment = {
      label: document.getElementById('builderSpecName')?.value || 'WebExperiment',
      local_dynamics: local_dynamics,
      network: network,
      integration: integration,
      coupling: coupling,
      monitors: monitors,
    };

    if (stimulus) {
      experiment.stimulus = stimulus;
    }

    log('Collected experiment:', experiment);
    return experiment;
  }

  function initializeRunTab() {
    log('Initializing Run tab');

    const runBtn = document.getElementById('runSimulationBtn');
    const plotTypeSelect = document.getElementById('plotType');
    const plotStateVarsContainer = document.getElementById('plotStateVars');
    const plotRegionsContainer = document.getElementById('plotRegions');
    const downloadBtn = document.getElementById('downloadResultsBtn');
    const selectAllBtn = document.getElementById('selectAllRegions');
    const selectNoneBtn = document.getElementById('selectNoneRegions');

    runBtn?.addEventListener('click', runSimulation);
    plotTypeSelect?.addEventListener('change', () => {
      updateStateVarHint();
      updatePlot();
    });
    // State variable checkboxes are handled via event delegation
    plotStateVarsContainer?.addEventListener('change', (e) => {
      enforceStateVarSelection(e.target);
      updatePlot();
    });
    // Region checkboxes are handled via event delegation
    plotRegionsContainer?.addEventListener('change', updatePlot);
    downloadBtn?.addEventListener('click', downloadResults);

    // Select All / Select None buttons
    selectAllBtn?.addEventListener('click', () => {
      const checkboxes = plotRegionsContainer?.querySelectorAll('input[type="checkbox"]');
      checkboxes?.forEach(cb => cb.checked = true);
      updatePlot();
    });
    selectNoneBtn?.addEventListener('click', () => {
      const checkboxes = plotRegionsContainer?.querySelectorAll('input[type="checkbox"]');
      checkboxes?.forEach(cb => cb.checked = false);
      updatePlot();
    });
  }

  function updateStateVarHint() {
    const plotType = document.getElementById('plotType')?.value;
    const hint = document.getElementById('stateVarHint');
    const plotTypeHint = document.getElementById('plotTypeHint');

    if (hint) {
      if (plotType === 'phasespace') {
        hint.textContent = 'Select exactly 2 variables for X and Y axes';
        hint.style.color = '#dc3545';
      } else if (plotType === 'heatmap') {
        hint.textContent = 'Select 1 variable for heatmap';
        hint.style.color = '#dc3545';
      } else {
        hint.textContent = 'Select variables to plot';
        hint.style.color = '#6c757d';
      }
    }
  }

  function enforceStateVarSelection(changedCheckbox) {
    const plotType = document.getElementById('plotType')?.value;
    const container = document.getElementById('plotStateVars');
    if (!container) return;

    const checkboxes = container.querySelectorAll('input.statevar-checkbox');
    const checked = Array.from(checkboxes).filter(cb => cb.checked);

    if (plotType === 'heatmap') {
      // Only 1 allowed - uncheck others when one is checked
      if (changedCheckbox.checked) {
        checkboxes.forEach(cb => {
          if (cb !== changedCheckbox) cb.checked = false;
        });
      }
    } else if (plotType === 'phasespace') {
      // Only 2 allowed - if more than 2, uncheck the oldest (first in DOM that isn't the current)
      if (checked.length > 2) {
        for (const cb of checkboxes) {
          if (cb.checked && cb !== changedCheckbox) {
            cb.checked = false;
            break;
          }
        }
      }
    }
    // timeseries: no limit
  }

  async function runSimulation() {
    console.log('[ModelBuilder] ========== SIMULATION START ==========');
    log('Running simulation...');

    const runBtn = document.getElementById('runSimulationBtn');
    const statusDiv = document.getElementById('runStatus');
    const statusText = document.getElementById('runStatusText');
    const progressBar = document.getElementById('runProgress');
    const errorDiv = document.getElementById('runError');
    const resultsDiv = document.getElementById('runResults');

    // Get run parameters
    const duration = parseFloat(document.getElementById('runDuration')?.value) || 1000;
    const stepSize = parseFloat(document.getElementById('runStepSize')?.value) || 0.1;
    const backend = document.getElementById('runBackend')?.value || 'jax';
    console.log('[ModelBuilder] Run parameters:', { duration, stepSize, backend });

    // Reset UI
    errorDiv.style.display = 'none';
    resultsDiv.style.display = 'none';
    statusDiv.style.display = 'block';
    runBtn.disabled = true;
    progressBar.style.width = '10%';
    statusText.textContent = 'Collecting experiment configuration...';

    try {
      // Collect the full experiment configuration
      console.log('[ModelBuilder] Collecting experiment configuration...');
      const experiment = collectFullExperiment();
      console.log('[ModelBuilder] Collected experiment:', JSON.stringify(experiment, null, 2));
      log('Experiment config:', experiment);

      progressBar.style.width = '20%';
      statusText.textContent = 'Sending to TVBO API...';

      // Call the Odoo endpoint which proxies to TVBO API
      const response = await fetch('/tvbo/configurator/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'call',
          params: {
            experiment: experiment,
            duration: duration,
            step_size: stepSize,
            backend: backend,
          },
          id: Date.now(),
        }),
      });

      progressBar.style.width = '80%';
      statusText.textContent = 'Processing results...';

      const result = await response.json();
      console.log('[ModelBuilder] Raw API response:', result);

      if (result.error) {
        console.error('[ModelBuilder] API returned error:', result.error);
        throw new Error(result.error.message || result.error.data?.message || 'API returned error');
      }

      const data = result.result;
      console.log('[ModelBuilder] Result data:', data);

      // MVP: Fail explicitly if data is missing
      if (!data) {
        throw new Error('API returned empty result. Check Odoo logs.');
      }
      if (!data.success) {
        console.error('[ModelBuilder] Simulation failed:', data.error);
        throw new Error(data.error);
      }
      if (!data.data) {
        throw new Error('Simulation returned no data array');
      }
      if (!data.time) {
        throw new Error('Simulation returned no time array');
      }
      if (!data.state_variables) {
        throw new Error('Simulation returned no state_variables');
      }

      console.log('[ModelBuilder] data.data length:', data.data.length);
      console.log('[ModelBuilder] data.time length:', data.time.length);
      console.log('[ModelBuilder] data.state_variables:', data.state_variables);
      console.log('[ModelBuilder] data.region_labels:', data.region_labels);

      // Store results - no fallbacks, fail if data missing
      simulationResults = {
        data: data.data,
        time: data.time,
        stateVariables: data.state_variables,
        regionLabels: data.region_labels,
        samplePeriod: data.sample_period,
      };
      console.log('[ModelBuilder] Stored simulationResults:', {
        dataShape: simulationResults.data ? `[${simulationResults.data.length}]` : 'null',
        timeLength: simulationResults.time?.length,
        stateVariables: simulationResults.stateVariables,
        regionLabels: simulationResults.regionLabels,
        samplePeriod: simulationResults.samplePeriod,
      });
      // Log first data point for debugging
      if (simulationResults.data && simulationResults.data.length > 0) {
        console.log('[ModelBuilder] First time point data:', simulationResults.data[0]);
        console.log('[ModelBuilder] Data structure: data[time][stateVar][region][mode]');
      }

      progressBar.style.width = '100%';
      statusText.textContent = 'Complete!';

      // Populate plot controls
      console.log('[ModelBuilder] Calling populatePlotControls...');
      populatePlotControls();

      // Show results and plot
      setTimeout(() => {
        statusDiv.style.display = 'none';
        resultsDiv.style.display = 'block';
        updatePlot();

        // Update info
        const infoDiv = document.getElementById('simInfo');
        if (infoDiv) {
          const nT = simulationResults.time?.length || 0;
          const nSV = simulationResults.stateVariables?.length || 0;
          const nR = simulationResults.regionLabels?.length || 0;
          infoDiv.textContent = `Duration: ${duration}ms | Step: ${stepSize}ms | Time points: ${nT} | State variables: ${nSV} | Regions: ${nR}`;
        }
      }, 500);

    } catch (err) {
      log('Simulation error:', err);
      statusDiv.style.display = 'none';
      errorDiv.style.display = 'block';
      errorDiv.textContent = `Error: ${err.message}`;
    } finally {
      runBtn.disabled = false;
    }
  }

  function populatePlotControls() {
    console.log('[ModelBuilder] populatePlotControls called, simulationResults:', simulationResults);
    if (!simulationResults) {
      throw new Error('populatePlotControls called but simulationResults is null');
    }

    const stateVarsContainer = document.getElementById('plotStateVars');
    const regionsContainer = document.getElementById('plotRegions');

    if (!stateVarsContainer) {
      throw new Error('plotStateVars container not found');
    }
    if (!regionsContainer) {
      throw new Error('plotRegions container not found');
    }

    // Populate state variables as checkboxes - first one selected by default
    stateVarsContainer.innerHTML = simulationResults.stateVariables
      .map((sv, i) => `
        <div class="form-check" style="margin-bottom: 2px;">
          <input class="form-check-input statevar-checkbox" type="checkbox" value="${i}" id="statevar_${i}" ${i === 0 ? 'checked' : ''}>
          <label class="form-check-label" for="statevar_${i}" style="font-size: 0.9em;">${escapeHtml(sv)}</label>
        </div>
      `)
      .join('');
    console.log('[ModelBuilder] State var checkboxes populated:', simulationResults.stateVariables);

    // Populate regions as checkboxes - generate labels if not provided by API
    const nRegions = simulationResults.data[0][0].length;
    const labels = simulationResults.regionLabels.length > 0
      ? simulationResults.regionLabels
      : Array.from({length: nRegions}, (_, i) => `Region_${i}`);
    console.log('[ModelBuilder] Region labels for checkboxes:', labels);

    regionsContainer.innerHTML = labels
      .map((label, i) => `
        <div class="form-check" style="margin-bottom: 2px;">
          <input class="form-check-input region-checkbox" type="checkbox" value="${i}" id="region_${i}" ${i < 5 ? 'checked' : ''}>
          <label class="form-check-label" for="region_${i}" style="font-size: 0.9em;">${escapeHtml(label)}</label>
        </div>
      `)
      .join('');

    // Update hint based on current plot type
    updateStateVarHint();
  }

  function updatePlot() {
    console.log('[ModelBuilder] updatePlot called, simulationResults:', simulationResults);
    if (!simulationResults) {
      throw new Error('updatePlot called but simulationResults is null');
    }
    if (!simulationResults.data) {
      throw new Error('updatePlot called but simulationResults.data is null');
    }
    if (!simulationResults.time) {
      throw new Error('updatePlot called but simulationResults.time is null');
    }

    const plotTypeSelect = document.getElementById('plotType');
    const stateVarsContainer = document.getElementById('plotStateVars');
    const regionsContainer = document.getElementById('plotRegions');
    const container = document.getElementById('plotContainer');

    if (!plotTypeSelect) throw new Error('plotType select not found');
    if (!stateVarsContainer) throw new Error('plotStateVars container not found');
    if (!regionsContainer) throw new Error('plotRegions container not found');
    if (!container) throw new Error('plotContainer not found');

    const plotType = plotTypeSelect.value;
    // Get selected state variables from checkboxes
    const selectedStateVars = Array.from(stateVarsContainer.querySelectorAll('input.statevar-checkbox:checked'))
      .map(cb => parseInt(cb.value));
    // Get selected regions from checkboxes
    const selectedRegions = Array.from(regionsContainer.querySelectorAll('input.region-checkbox:checked'))
      .map(cb => parseInt(cb.value));

    console.log('[ModelBuilder] Plot settings:', { plotType, selectedStateVars, selectedRegions });
    console.log('[ModelBuilder] Plot container dimensions:', container.clientWidth, 'x', container.clientHeight);

    const time = simulationResults.time;
    const data = simulationResults.data;

    // Validate selections based on plot type
    if (plotType === 'timeseries') {
      if (selectedStateVars.length === 0) {
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#6c757d;">Please select at least one state variable</div>';
        return;
      }
      plotTimeSeries(container, time, data, selectedStateVars, selectedRegions);
    } else if (plotType === 'phasespace') {
      if (selectedStateVars.length !== 2) {
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#6c757d;">Please select exactly 2 state variables for phase space plot</div>';
        return;
      }
      plotPhaseSpace(container, data, selectedStateVars, selectedRegions);
    } else if (plotType === 'heatmap') {
      if (selectedStateVars.length !== 1) {
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#6c757d;">Please select exactly 1 state variable for heatmap</div>';
        return;
      }
      plotHeatmap(container, time, data, selectedStateVars[0]);
    }
  }

  function plotTimeSeries(container, time, data, stateVarIndices, regions) {
    // Create SVG-based time series plot
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 20, right: 120, bottom: 40, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Extract data for selected state variables and regions
    const traces = [];
    stateVarIndices.forEach(stateVarIdx => {
      regions.forEach(regionIdx => {
        const values = time.map((_, tIdx) => {
          // data shape: [time, state_vars, regions, modes]
          const val = data[tIdx]?.[stateVarIdx]?.[regionIdx]?.[0];
          return val !== undefined ? val : 0;
        });
        const svName = simulationResults.stateVariables[stateVarIdx] || `sv${stateVarIdx}`;
        const regionLabel = simulationResults.regionLabels[regionIdx] || `Region_${regionIdx}`;
        traces.push({ stateVarIdx, regionIdx, values, label: `${svName} - ${regionLabel}` });
      });
    });

    // Find data range
    const allValues = traces.flatMap(t => t.values);
    const yMin = Math.min(...allValues);
    const yMax = Math.max(...allValues);
    const yRange = yMax - yMin || 1;

    const xScale = (t) => margin.left + (t / time[time.length - 1]) * plotWidth;
    const yScale = (v) => margin.top + plotHeight - ((v - yMin) / yRange) * plotHeight;

    // Colors for different regions
    const colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#66c2a5', '#fc8d62'];

    let svg = `<svg width="${width}" height="${height}" style="background: white;">`;

    // Grid lines
    for (let i = 0; i <= 5; i++) {
      const y = margin.top + (plotHeight / 5) * i;
      const yVal = yMax - (yRange / 5) * i;
      svg += `<line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" stroke="#e0e0e0" stroke-width="1"/>`;
      svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" font-size="11" fill="#666">${yVal.toFixed(2)}</text>`;
    }

    // X axis labels
    const numXLabels = 5;
    for (let i = 0; i <= numXLabels; i++) {
      const tVal = (time[time.length - 1] / numXLabels) * i;
      const x = xScale(tVal);
      svg += `<text x="${x}" y="${height - 10}" text-anchor="middle" font-size="11" fill="#666">${tVal.toFixed(0)} ms</text>`;
    }

    // Plot traces
    traces.forEach((trace, idx) => {
      const color = colors[idx % colors.length];
      const points = trace.values.map((v, i) => `${xScale(time[i])},${yScale(v)}`).join(' ');
      svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1.5"/>`;

      // Legend
      const legendY = margin.top + idx * 16;
      svg += `<rect x="${width - margin.right + 10}" y="${legendY}" width="10" height="10" fill="${color}"/>`;
      svg += `<text x="${width - margin.right + 24}" y="${legendY + 9}" font-size="10" fill="#333">${escapeHtml(trace.label)}</text>`;
    });

    // Axes
    svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#333" stroke-width="1"/>`;
    svg += `<line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#333" stroke-width="1"/>`;

    // Axis labels
    svg += `<text x="${margin.left / 2}" y="${height / 2}" text-anchor="middle" transform="rotate(-90, ${margin.left / 2}, ${height / 2})" font-size="12" fill="#333">Value</text>`;
    svg += `<text x="${margin.left + plotWidth / 2}" y="${height - 2}" text-anchor="middle" font-size="12" fill="#333">Time (ms)</text>`;

    svg += '</svg>';
    container.innerHTML = svg;
  }

  function plotPhaseSpace(container, data, stateVarIndices, regions) {
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 20, right: 100, bottom: 40, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Need exactly 2 state variables for phase space
    if (stateVarIndices.length !== 2) {
      container.innerHTML = '<div class="alert alert-warning">Phase space plot requires exactly 2 state variables.</div>';
      return;
    }

    const sv0 = stateVarIndices[0];
    const sv1 = stateVarIndices[1];
    const colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#66c2a5', '#fc8d62', '#8da0cb'];

    // Extract the two selected state variables for all selected regions
    const traces = regions.map(regionIdx => {
      const xVals = data.map(t => t[sv0]?.[regionIdx]?.[0] || 0);
      const yVals = data.map(t => t[sv1]?.[regionIdx]?.[0] || 0);
      const regionLabel = simulationResults.regionLabels[regionIdx] || `Region_${regionIdx}`;
      return { regionIdx, xVals, yVals, label: regionLabel };
    });

    const allX = traces.flatMap(t => t.xVals);
    const allY = traces.flatMap(t => t.yVals);
    const xMin = Math.min(...allX), xMax = Math.max(...allX);
    const yMin = Math.min(...allY), yMax = Math.max(...allY);
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const xScale = (v) => margin.left + ((v - xMin) / xRange) * plotWidth;
    const yScale = (v) => margin.top + plotHeight - ((v - yMin) / yRange) * plotHeight;

    let svg = `<svg width="${width}" height="${height}" style="background: white;">`;

    // Plot traces
    traces.forEach((trace, idx) => {
      const color = colors[idx % colors.length];
      const points = trace.xVals.map((x, i) => `${xScale(x)},${yScale(trace.yVals[i])}`).join(' ');
      svg += `<polyline points="${points}" fill="none" stroke="${color}" stroke-width="1" opacity="0.8"/>`;

      // Legend
      const legendY = margin.top + idx * 16;
      svg += `<rect x="${width - margin.right + 10}" y="${legendY}" width="10" height="10" fill="${color}"/>`;
      svg += `<text x="${width - margin.right + 24}" y="${legendY + 9}" font-size="10" fill="#333">${escapeHtml(trace.label)}</text>`;
    });

    // Axes
    svg += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" stroke="#333" stroke-width="1"/>`;
    svg += `<line x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" stroke="#333" stroke-width="1"/>`;

    // Labels using selected state variable names
    const sv1Name = simulationResults.stateVariables?.[sv1] || 'Y';
    const sv0Name = simulationResults.stateVariables?.[sv0] || 'X';
    svg += `<text x="${margin.left / 2}" y="${height / 2}" text-anchor="middle" transform="rotate(-90, ${margin.left / 2}, ${height / 2})" font-size="12">${sv1Name}</text>`;
    svg += `<text x="${margin.left + plotWidth / 2}" y="${height - 5}" text-anchor="middle" font-size="12">${sv0Name}</text>`;

    svg += '</svg>';
    container.innerHTML = svg;
  }

  function plotHeatmap(container, time, data, stateVarIdx) {
    const width = container.clientWidth;
    const height = container.clientHeight;
    const margin = { top: 20, right: 100, bottom: 40, left: 80 };

    const nTime = time.length;
    const nRegions = data[0]?.[stateVarIdx]?.length || 0;

    if (nTime === 0 || nRegions === 0) {
      container.innerHTML = '<div class="alert alert-warning">No data for heatmap.</div>';
      return;
    }

    // Downsample if too many time points
    const maxTimePoints = 500;
    const timeStep = Math.max(1, Math.floor(nTime / maxTimePoints));
    const sampledTime = time.filter((_, i) => i % timeStep === 0);
    const sampledData = data.filter((_, i) => i % timeStep === 0);

    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const cellWidth = plotWidth / sampledTime.length;
    const cellHeight = plotHeight / nRegions;

    // Find data range
    let vMin = Infinity, vMax = -Infinity;
    sampledData.forEach(t => {
      for (let r = 0; r < nRegions; r++) {
        const v = t[stateVarIdx]?.[r]?.[0] || 0;
        vMin = Math.min(vMin, v);
        vMax = Math.max(vMax, v);
      }
    });
    const vRange = vMax - vMin || 1;

    // Color function (blue-white-red)
    const colorScale = (v) => {
      const norm = (v - vMin) / vRange;
      if (norm < 0.5) {
        const t = norm * 2;
        return `rgb(${Math.round(59 + 196 * t)}, ${Math.round(76 + 179 * t)}, ${Math.round(192 - 47 * t)})`;
      } else {
        const t = (norm - 0.5) * 2;
        return `rgb(255, ${Math.round(255 - 155 * t)}, ${Math.round(145 - 115 * t)})`;
      }
    };

    let svg = `<svg width="${width}" height="${height}" style="background: white;">`;

    // Draw cells
    sampledData.forEach((t, tIdx) => {
      for (let r = 0; r < nRegions; r++) {
        const v = t[stateVarIdx]?.[r]?.[0] || 0;
        const x = margin.left + tIdx * cellWidth;
        const y = margin.top + r * cellHeight;
        svg += `<rect x="${x}" y="${y}" width="${cellWidth + 0.5}" height="${cellHeight + 0.5}" fill="${colorScale(v)}"/>`;
      }
    });

    // Y axis labels (regions)
    const maxLabels = 20;
    const labelStep = Math.max(1, Math.floor(nRegions / maxLabels));
    for (let r = 0; r < nRegions; r += labelStep) {
      const y = margin.top + r * cellHeight + cellHeight / 2;
      const label = simulationResults.regionLabels[r] || `${r}`;
      svg += `<text x="${margin.left - 5}" y="${y + 4}" text-anchor="end" font-size="10" fill="#333">${label}</text>`;
    }

    // X axis labels (time)
    for (let i = 0; i <= 4; i++) {
      const tIdx = Math.floor(i * (sampledTime.length - 1) / 4);
      const x = margin.left + tIdx * cellWidth;
      svg += `<text x="${x}" y="${height - 10}" text-anchor="middle" font-size="10" fill="#333">${sampledTime[tIdx]?.toFixed(0) || 0} ms</text>`;
    }

    // Colorbar
    const cbX = width - margin.right + 20;
    const cbHeight = plotHeight;
    for (let i = 0; i < 50; i++) {
      const v = vMax - (i / 49) * vRange;
      const y = margin.top + (i / 49) * cbHeight;
      svg += `<rect x="${cbX}" y="${y}" width="20" height="${cbHeight / 49 + 1}" fill="${colorScale(v)}"/>`;
    }
    svg += `<text x="${cbX + 25}" y="${margin.top + 10}" font-size="10">${vMax.toFixed(2)}</text>`;
    svg += `<text x="${cbX + 25}" y="${margin.top + cbHeight}" font-size="10">${vMin.toFixed(2)}</text>`;

    svg += '</svg>';
    container.innerHTML = svg;
  }

  function downloadResults() {
    if (!simulationResults) {
      alert('No simulation results to download.');
      return;
    }

    // Convert to CSV
    const time = simulationResults.time;
    const data = simulationResults.data;
    const stateVars = simulationResults.stateVariables;
    const regions = simulationResults.regionLabels;
    const nRegions = data[0]?.[0]?.length || 0;

    // Header
    let csv = 'time';
    for (let sv = 0; sv < stateVars.length; sv++) {
      for (let r = 0; r < nRegions; r++) {
        const regionLabel = regions[r] || `region_${r}`;
        csv += `,${stateVars[sv]}_${regionLabel}`;
      }
    }
    csv += '\n';

    // Data rows
    for (let t = 0; t < time.length; t++) {
      csv += time[t].toFixed(4);
      for (let sv = 0; sv < stateVars.length; sv++) {
        for (let r = 0; r < nRegions; r++) {
          const val = data[t]?.[sv]?.[r]?.[0] || 0;
          csv += `,${val}`;
        }
      }
      csv += '\n';
    }

    // Download
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'simulation_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  }


  // Initialize all tabs on DOM ready
  document.addEventListener('DOMContentLoaded', function() {
    // Initialize tab content when tabs are clicked
    document.getElementById('network-tab')?.addEventListener('shown.bs.tab', function() {
      initializeNetworkTab();
      initializeCouplingTab();
    });
    document.getElementById('integration-tab')?.addEventListener('shown.bs.tab', function() {
      initializeIntegratorTab();
    });
    document.getElementById('observation-tab')?.addEventListener('shown.bs.tab', function() {
      initializeObservationModelsTab();
    });
    document.getElementById('stimulus-tab')?.addEventListener('shown.bs.tab', function() {
      initializeStimulusTab();
    });
    document.getElementById('preview-tab')?.addEventListener('shown.bs.tab', function() {
      initializePreviewTab();
    });
    document.getElementById('run-tab')?.addEventListener('shown.bs.tab', function() {
      initializeRunTab();
    });
  });
})();
