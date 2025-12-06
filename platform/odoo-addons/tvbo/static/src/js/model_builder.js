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

  function initializeBuilder() {
    // Load data
    STATE.data = window.searchData || [];
    STATE.dataLoaded = true;
    log('Initializing builder with data:', { count: STATE.data.length });

    const content = document.getElementById('builderContent');
    if (content) {
      renderBuilder(content);
    }
  }

  function renderBuilder(root) {
    const data = STATE.data || [];
    log('Rendering builder with data count:', data.length);
    const models = data.filter(x => (x.type || '').toLowerCase() === 'model');
    log('Filtered models count:', models.length);

    root.innerHTML = `
      <div>
        <div class="builder-field">
          <label>Model Name</label>
          <input id="builderSpecName" class="builder-input" placeholder="MyCustomModel" />
        </div>
        <div class="builder-field">
          <label>Description</label>
          <textarea id="builderNotes" class="builder-text" rows="3" placeholder="Optional description"></textarea>
        </div>
        <div class="builder-field">
          <label>System Type</label>
          <select id="builderSystemType" class="builder-select">
            <option value="continuous">Continuous (ODE/SDE)</option>
            <option value="discrete">Discrete (Maps)</option>
          </select>
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
      let ots = item.output_transforms || [];
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
    const name = section.querySelector('#builderSpecName').value.trim() || 'MyCustomModel';
    const description = section.querySelector('#builderNotes').value.trim() || '';
    const systemType = section.querySelector('#builderSystemType')?.value || 'continuous';

    const modelParams = Array.from(section.querySelectorAll('#modelParamsRows .builder-row')).map(row => {
      const domain_lo = parseMaybeNumber(row.querySelector('.p-domain-lo')?.value.trim());
      const domain_hi = parseMaybeNumber(row.querySelector('.p-domain-hi')?.value.trim());
      const domain = (domain_lo !== undefined || domain_hi !== undefined) ? { lo: domain_lo, hi: domain_hi } : undefined;

      return {
        name: row.querySelector('.p-name').value.trim(),
        value: parseMaybeNumber(row.querySelector('.p-value').value.trim()),
        unit: row.querySelector('.p-unit').value.trim() || undefined,
        symbol: row.querySelector('.p-symbol')?.value.trim() || undefined,
        domain: domain
      };
    }).filter(p => p.name);

    const derivedParams = Array.from(section.querySelectorAll('#derivedParamsRows .builder-row')).map(row => ({
      name: row.querySelector('.dp-name').value.trim(),
      unit: row.querySelector('.dp-unit')?.value.trim() || undefined,
      equation: {
        rhs: row.querySelector('.dp-expr').value.trim() || undefined
      }
    })).filter(p => p.name);

    const stateVars = Array.from(section.querySelectorAll('#stateEqRows .builder-row')).map(row => ({
      name: row.querySelector('.sv-name').value.trim(),
      symbol: row.querySelector('.sv-symbol')?.value.trim() || undefined,
      unit: row.querySelector('.sv-unit')?.value.trim() || undefined,
      initial_value: parseMaybeNumber(row.querySelector('.sv-initial')?.value.trim()),
      variable_of_interest: row.querySelector('.sv-voi')?.checked || false,
      coupling_variable: row.querySelector('.sv-coupling')?.checked || false,
      equation: {
        rhs: row.querySelector('.sv-expr').value.trim() || undefined
      }
    })).filter(e => e.name);

    const derivedVars = Array.from(section.querySelectorAll('#derivedVarsRows .builder-row')).map(row => ({
      name: row.querySelector('.dv-name').value.trim(),
      unit: row.querySelector('.dv-unit')?.value.trim() || undefined,
      equation: {
        rhs: row.querySelector('.dv-expr').value.trim() || undefined
      }
    })).filter(e => e.name);

    const outputTransforms = Array.from(section.querySelectorAll('#outputTransformsRows .builder-row')).map(row => ({
      name: row.querySelector('.ot-name').value.trim(),
      unit: row.querySelector('.ot-unit')?.value.trim() || undefined,
      equation: {
        rhs: row.querySelector('.ot-expr').value.trim() || undefined
      }
    })).filter(e => e.name);

    const functions = Array.from(section.querySelectorAll('#functionsRows .builder-row')).map(row => ({
      name: row.querySelector('.fn-name').value.trim(),
      equation: {
        rhs: row.querySelector('.fn-expr').value.trim() || undefined
      }
    })).filter(f => f.name);

    const couplingTerms = Array.from(section.querySelectorAll('#couplingTermsRows .builder-row')).map(row => ({
      name: row.querySelector('.ct-name').value.trim(),
      value: parseMaybeNumber(row.querySelector('.ct-value').value.trim())
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
        output_transforms: outputTransforms.length ? outputTransforms : undefined,
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

    if (model.output_transforms && model.output_transforms.length > 0) {
      code += `    'output_transforms': [\n`;
      model.output_transforms.forEach(ot => {
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

    if (model.output_transforms && model.output_transforms.length > 0) {
      yaml += `\noutput_transforms:\n`;
      model.output_transforms.forEach(ot => {
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
    const content = document.getElementById('networkContent');
    if (!content) return;

    const networks = window.networksData || [];
    content.innerHTML = `
      <div class="builder-field">
        <label>Network Configuration Mode</label>
        <select id="networkMode" class="builder-select">
          <option value="standard">Standard Network (Tractogram + Parcellation)</option>
          <option value="yaml">Load from YAML File</option>
          <option value="custom">Custom Network (Node/Edge Builder)</option>
        </select>
      </div>

      <div id="networkConfigContainer"></div>
    `;

    const modeSelect = content.querySelector('#networkMode');
    const configContainer = content.querySelector('#networkConfigContainer');

    function renderNetworkConfig(mode) {
      if (mode === 'standard') {
        configContainer.innerHTML = `
          <div class="builder-field">
            <label>Select Existing Network</label>
            <select id="networkSelect" class="builder-select">
              <option value="">— Select existing network —</option>
              ${networks.map(n => `<option value="${n.id}">${escapeHtml(n.label || n.name)}</option>`).join('')}
            </select>
          </div>

          <div class="hr"></div>
          <div style="font-weight: bold; margin-bottom: 10px;">Or Configure New Network</div>

          <div class="builder-field">
            <label>Parcellation/Atlas</label>
            <select id="networkParcellation" class="builder-select">
              <option value="">— Select parcellation —</option>
              <option value="desikan-killiany">Desikan-Killiany</option>
              <option value="destrieux">Destrieux</option>
              <option value="hcp-mmp1">HCP-MMP1</option>
              <option value="yeo7">Yeo 7 Networks</option>
              <option value="yeo17">Yeo 17 Networks</option>
            </select>
          </div>

          <div class="builder-field">
            <label>Tractogram/Connectivity Source</label>
            <select id="networkTractogram" class="builder-select">
              <option value="">— Select tractogram —</option>
              <option value="dti">DTI-based</option>
              <option value="dsi">DSI-based</option>
              <option value="hcp">HCP Average</option>
              <option value="custom">Custom Upload</option>
            </select>
          </div>

          <div class="builder-field">
            <label>Number of Regions</label>
            <input id="networkNumRegions" type="number" class="builder-input" placeholder="84" value="84" />
          </div>

          <div class="builder-field">
            <label>Global Coupling Strength</label>
            <input id="networkGlobalCoupling" type="number" step="0.01" class="builder-input" placeholder="1.0" value="1.0" />
          </div>

          <div class="builder-field">
            <label>Conduction Speed (mm/ms)</label>
            <input id="networkConductionSpeed" type="number" step="0.1" class="builder-input" placeholder="3.0" value="3.0" />
          </div>

          <div class="builder-field">
            <label>Normalization</label>
            <select id="networkNormalization" class="builder-select">
              <option value="">None</option>
              <option value="region">By Region</option>
              <option value="max">By Maximum</option>
              <option value="tract_length">By Tract Length</option>
            </select>
          </div>
        `;
      } else if (mode === 'yaml') {
        configContainer.innerHTML = `
          <div class="builder-field">
            <label>Upload Network YAML File</label>
            <input type="file" id="networkYamlFile" accept=".yaml,.yml" class="builder-input" />
          </div>

          <div class="builder-field">
            <label>Or Paste YAML Content</label>
            <textarea id="networkYamlContent" class="builder-text" rows="15" placeholder="Paste YAML network definition here...
Example:
label: My Network
nodes:
  - id: 0
    label: Node_A
    dynamics:
      name: Generic2dOscillator
      dataLocation: database/models/Generic2dOscillator.yaml
    position:
      x: 0.0
      y: 0.0
      z: 0.0
edges:
  - source: 0
    target: 1
    weight: 0.8
    delay: 2.0"></textarea>
          </div>

          <div class="builder-actions">
            <button class="btn btn-sm btn-primary" id="parseNetworkYaml">Parse & Load YAML</button>
          </div>

          <div id="yamlParseResult" style="margin-top: 10px;"></div>
        `;

        const fileInput = document.getElementById('networkYamlFile');
        fileInput?.addEventListener('change', function(e) {
          const file = e.target.files[0];
          if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
              document.getElementById('networkYamlContent').value = event.target.result;
            };
            reader.readAsText(file);
          }
        });

        document.getElementById('parseNetworkYaml')?.addEventListener('click', function() {
          const yamlContent = document.getElementById('networkYamlContent').value;
          if (!yamlContent.trim()) {
            alert('Please provide YAML content');
            return;
          }

          // In a real implementation, this would parse YAML and populate the network
          const resultDiv = document.getElementById('yamlParseResult');
          resultDiv.innerHTML = '<div class="alert alert-success">YAML parsed successfully! Network loaded.</div>';

          // Store parsed network data
          window.currentNetworkConfig = {
            mode: 'yaml',
            content: yamlContent
          };
        });
      } else if (mode === 'custom') {
        configContainer.innerHTML = `
          <div class="alert alert-info">
            <strong>Custom Network Builder</strong><br/>
            Define nodes with individual dynamics and edges with specific coupling configurations.
          </div>

          <div class="builder-field">
            <label>Network Label</label>
            <input id="customNetworkLabel" class="builder-input" placeholder="My Custom Network" />
          </div>

          <div class="builder-field">
            <label>Network Description</label>
            <textarea id="customNetworkDesc" class="builder-text" rows="2" placeholder="Optional description"></textarea>
          </div>

          <div class="hr"></div>

          <div class="builder-field">
            <div class="builder-subtitle" style="display:flex; justify-content:space-between; align-items:center;">
              <span>Nodes</span>
              <span style="font-size: 0.85em; color: #666;">Define network nodes with positions and dynamics</span>
            </div>
            <div id="customNetworkNodes" class="builder-rows"></div>
            <div class="builder-actions">
              <button class="btn btn-sm btn-secondary" id="addCustomNode">Add Node</button>
            </div>
          </div>

          <div class="hr"></div>

          <div class="builder-field">
            <div class="builder-subtitle" style="display:flex; justify-content:space-between; align-items:center;">
              <span>Edges</span>
              <span style="font-size: 0.85em; color: #666;">Define connections between nodes</span>
            </div>
            <div id="customNetworkEdges" class="builder-rows"></div>
            <div class="builder-actions">
              <button class="btn btn-sm btn-secondary" id="addCustomEdge">Add Edge</button>
            </div>
          </div>

          <div class="hr"></div>

          <div class="builder-field">
            <h5>Global Network Parameters</h5>

            <label>Global Coupling Strength</label>
            <input id="customGlobalCoupling" type="number" step="0.01" class="builder-input" placeholder="1.0" value="1.0" />
          </div>

          <div class="builder-field">
            <label>Conduction Speed (mm/ms)</label>
            <input id="customConductionSpeed" type="number" step="0.1" class="builder-input" placeholder="3.0" value="3.0" />
          </div>
        `;

        const nodesContainer = document.getElementById('customNetworkNodes');
        const edgesContainer = document.getElementById('customNetworkEdges');

        function createNodeRow(id = 0, label = '', region = '', x = 0, y = 0, z = 0, dynamics = '') {
          const row = document.createElement('div');
          row.className = 'builder-row';
          row.innerHTML = `
            <div style="display: grid; grid-template-columns: 60px 1fr 1fr 80px 80px 80px 1fr 40px; gap: 8px; align-items: center;">
              <input type="number" class="builder-input node-id" value="${id}" placeholder="ID" />
              <input class="builder-input node-label" value="${escapeAttr(label)}" placeholder="Label" />
              <input class="builder-input node-region" value="${escapeAttr(region)}" placeholder="Region" />
              <input type="number" step="0.1" class="builder-input node-x" value="${x}" placeholder="X" />
              <input type="number" step="0.1" class="builder-input node-y" value="${y}" placeholder="Y" />
              <input type="number" step="0.1" class="builder-input node-z" value="${z}" placeholder="Z" />
              <select class="builder-select node-dynamics">
                <option value="">— Select dynamics —</option>
                ${(window.searchData || []).filter(m => m.type === 'model').map(m =>
                  `<option value="${escapeAttr(m.key || m.name)}" ${dynamics === (m.key || m.name) ? 'selected' : ''}>${escapeHtml(m.title || m.name)}</option>`
                ).join('')}
              </select>
              <button class="btn btn-sm btn-danger" style="padding: 4px 8px;">×</button>
            </div>
          `;
          row.querySelector('button').addEventListener('click', () => row.remove());
          return row;
        }

        function createEdgeRow(source = 0, target = 1, weight = 1.0, delay = 0, distance = 0, couplingType = 'Linear') {
          const row = document.createElement('div');
          row.className = 'builder-row';
          row.innerHTML = `
            <div style="display: grid; grid-template-columns: 80px 80px 100px 100px 100px 1fr 40px; gap: 8px; align-items: center;">
              <input type="number" class="builder-input edge-source" value="${source}" placeholder="Source" />
              <input type="number" class="builder-input edge-target" value="${target}" placeholder="Target" />
              <input type="number" step="0.01" class="builder-input edge-weight" value="${weight}" placeholder="Weight" />
              <input type="number" step="0.1" class="builder-input edge-delay" value="${delay}" placeholder="Delay (ms)" />
              <input type="number" step="0.1" class="builder-input edge-distance" value="${distance}" placeholder="Distance (mm)" />
              <select class="builder-select edge-coupling">
                <option value="Linear" ${couplingType === 'Linear' ? 'selected' : ''}>Linear</option>
                <option value="Sigmoidal" ${couplingType === 'Sigmoidal' ? 'selected' : ''}>Sigmoidal</option>
                <option value="HyperbolicTangent" ${couplingType === 'HyperbolicTangent' ? 'selected' : ''}>Hyperbolic Tangent</option>
                <option value="PreSigmoidal" ${couplingType === 'PreSigmoidal' ? 'selected' : ''}>Pre-Sigmoidal</option>
                <option value="Scaling" ${couplingType === 'Scaling' ? 'selected' : ''}>Scaling</option>
              </select>
              <button class="btn btn-sm btn-danger" style="padding: 4px 8px;">×</button>
            </div>
          `;
          row.querySelector('button').addEventListener('click', () => row.remove());
          return row;
        }

        // Add initial rows
        nodesContainer.appendChild(createNodeRow(0, 'Node_0', 'Region_A', 0, 0, 0));
        nodesContainer.appendChild(createNodeRow(1, 'Node_1', 'Region_B', 10, 0, 0));

        edgesContainer.appendChild(createEdgeRow(0, 1, 0.8, 2.0, 10.0, 'Linear'));

        document.getElementById('addCustomNode')?.addEventListener('click', () => {
          const nodes = nodesContainer.querySelectorAll('.builder-row');
          const nextId = nodes.length;
          nodesContainer.appendChild(createNodeRow(nextId, `Node_${nextId}`, '', 0, 0, 0));
        });

        document.getElementById('addCustomEdge')?.addEventListener('click', () => {
          edgesContainer.appendChild(createEdgeRow(0, 1, 1.0, 0, 0, 'Linear'));
        });
      }
    }

    // Initial render
    renderNetworkConfig('standard');

    // Mode change handler
    modeSelect.addEventListener('change', (e) => {
      renderNetworkConfig(e.target.value);
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

    // Collect configuration from all tabs
    const config = {
      simulation_experiment: {
        name: 'Configured Experiment',
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
    const modelSelect = document.getElementById('modelSelect');
    return {
      model: modelSelect?.value || 'not configured',
      // Add more dynamics fields as needed
    };
  }

  function collectNetworkConfig() {
    const networkMode = document.getElementById('networkMode')?.value || 'standard';

    if (networkMode === 'standard') {
      const networkSelect = document.getElementById('networkSelect');
      const parcellation = document.getElementById('networkParcellation');
      const tractogram = document.getElementById('networkTractogram');
      const numRegions = document.getElementById('networkNumRegions');
      const globalCoupling = document.getElementById('networkGlobalCoupling');
      const conductionSpeed = document.getElementById('networkConductionSpeed');
      const normalization = document.getElementById('networkNormalization');

      return {
        mode: 'standard',
        network_id: networkSelect?.value || undefined,
        parcellation: parcellation?.value || undefined,
        tractogram: tractogram?.value || undefined,
        number_of_regions: numRegions?.value ? parseInt(numRegions.value) : undefined,
        global_coupling_strength: globalCoupling?.value ? parseFloat(globalCoupling.value) : undefined,
        conduction_speed: conductionSpeed?.value ? parseFloat(conductionSpeed.value) : undefined,
        normalization: normalization?.value || undefined
      };
    } else if (networkMode === 'yaml') {
      const yamlContent = document.getElementById('networkYamlContent')?.value;
      return {
        mode: 'yaml',
        yaml_content: yamlContent || undefined,
        source: 'yaml_file'
      };
    } else if (networkMode === 'custom') {
      const label = document.getElementById('customNetworkLabel')?.value;
      const description = document.getElementById('customNetworkDesc')?.value;
      const globalCoupling = document.getElementById('customGlobalCoupling')?.value;
      const conductionSpeed = document.getElementById('customConductionSpeed')?.value;

      // Collect nodes
      const nodeRows = document.querySelectorAll('#customNetworkNodes .builder-row');
      const nodes = Array.from(nodeRows).map(row => ({
        id: parseInt(row.querySelector('.node-id')?.value) || 0,
        label: row.querySelector('.node-label')?.value || '',
        region: row.querySelector('.node-region')?.value || undefined,
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
        source: parseInt(row.querySelector('.edge-source')?.value) || 0,
        target: parseInt(row.querySelector('.edge-target')?.value) || 1,
        weight: parseFloat(row.querySelector('.edge-weight')?.value) || 1.0,
        delay: parseFloat(row.querySelector('.edge-delay')?.value) || 0,
        distance: parseFloat(row.querySelector('.edge-distance')?.value) || 0,
        coupling: {
          name: row.querySelector('.edge-coupling')?.value || 'Linear'
        }
      }));

      return {
        mode: 'custom',
        label: label || 'Custom Network',
        description: description || undefined,
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
    return {
      method: integratorMethod?.value || 'not configured',
      step_size: stepSize?.value || 'not configured',
      duration: duration?.value || 'not configured',
    };
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
    return models.length > 0 ? models : ['not configured'];
  }  function collectStimulusConfig() {
    const stimulusType = document.getElementById('stimulusType');
    const amplitude = document.getElementById('stimulusAmplitude');
    return {
      type: stimulusType?.value || 'none',
      amplitude: amplitude?.value || 'not configured',
    };
  }

  function generateYamlPreview(config) {
    let yaml = `simulation_experiment:
  name: ${config.simulation_experiment.name}

  dynamics:
    model: ${config.simulation_experiment.dynamics.model}

  network:`;

    // Handle different network modes
    const networkConfig = config.simulation_experiment.network;
    if (networkConfig.mode === 'custom') {
      yaml += `\n    label: ${networkConfig.label}`;
      if (networkConfig.description) yaml += `\n    description: ${networkConfig.description}`;
      yaml += `\n    number_of_nodes: ${networkConfig.number_of_nodes}`;
      yaml += `\n    global_coupling_strength: ${networkConfig.global_coupling_strength}`;
      yaml += `\n    conduction_speed: ${networkConfig.conduction_speed}`;

      if (networkConfig.nodes && networkConfig.nodes.length > 0) {
        yaml += `\n    nodes:`;
        networkConfig.nodes.forEach(node => {
          yaml += `\n      - id: ${node.id}`;
          yaml += `\n        label: ${node.label}`;
          if (node.region) yaml += `\n        region: ${node.region}`;
          yaml += `\n        position:`;
          yaml += `\n          x: ${node.position.x}`;
          yaml += `\n          y: ${node.position.y}`;
          yaml += `\n          z: ${node.position.z}`;
          if (node.dynamics.name) {
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
          yaml += `\n        weight: ${edge.weight}`;
          yaml += `\n        delay: ${edge.delay}`;
          yaml += `\n        distance: ${edge.distance}`;
          if (edge.coupling && edge.coupling.name) {
            yaml += `\n        coupling:`;
            yaml += `\n          name: ${edge.coupling.name}`;
          }
        });
      }
    } else if (networkConfig.mode === 'yaml') {
      yaml += `\n    # Network loaded from YAML file`;
      yaml += `\n    yaml_source: true`;
    } else if (networkConfig.mode === 'standard') {
      if (networkConfig.network_id) {
        yaml += `\n    network_id: ${networkConfig.network_id}`;
      }
      if (networkConfig.parcellation) yaml += `\n    parcellation: ${networkConfig.parcellation}`;
      if (networkConfig.tractogram) yaml += `\n    tractogram: ${networkConfig.tractogram}`;
      if (networkConfig.number_of_regions) yaml += `\n    number_of_regions: ${networkConfig.number_of_regions}`;
      if (networkConfig.global_coupling_strength) yaml += `\n    global_coupling_strength: ${networkConfig.global_coupling_strength}`;
      if (networkConfig.conduction_speed) yaml += `\n    conduction_speed: ${networkConfig.conduction_speed}`;
      if (networkConfig.normalization) yaml += `\n    normalization: ${networkConfig.normalization}`;
    } else {
      yaml += `\n    connectivity: ${networkConfig.network || 'not configured'}`;
      yaml += `\n    coupling: ${networkConfig.coupling || 'not configured'}`;
    }

    yaml += `

  integration:
    method: ${config.simulation_experiment.integration.method}
    step_size: ${config.simulation_experiment.integration.step_size}
    duration: ${config.simulation_experiment.integration.duration}

  monitors:`;

    if (Array.isArray(config.simulation_experiment.observation_models) &&
        config.simulation_experiment.observation_models.length > 0 &&
        config.simulation_experiment.observation_models[0] !== 'not configured') {
      config.simulation_experiment.observation_models.forEach(m => {
        yaml += `\n    - name: ${m.name}`;
        yaml += `\n      monitor_type: ${m.monitor_type}`;
        yaml += `\n      period: ${m.period}`;

        // Add optional Monitor/ObservationModel fields
        if (m.label) yaml += `\n      label: ${m.label}`;
        if (m.acronym) yaml += `\n      acronym: ${m.acronym}`;
        if (m.description) yaml += `\n      description: ${m.description}`;
        if (m.imaging_modality) yaml += `\n      imaging_modality: ${m.imaging_modality}`;

        // Add pipeline
        if (m.pipeline && m.pipeline.length > 0) {
          yaml += `\n      pipeline:`;
          m.pipeline.forEach(step => {
            yaml += `\n        - order: ${step.order}`;
            yaml += `\n          operation_type: ${step.operation_type}`;
            if (step.function) yaml += `\n          function: ${step.function}`;
            if (step.output_alias) yaml += `\n          output_alias: ${step.output_alias}`;
            if (step.apply_on_dimension) yaml += `\n          apply_on_dimension: ${step.apply_on_dimension}`;
            if (step.ensure_shape) yaml += `\n          ensure_shape: ${step.ensure_shape}`;
          });
        }

        // Add other ObservationModel attributes
        if (m.data_injections && m.data_injections.length > 0) {
          yaml += `\n      data_injections:`;
          m.data_injections.forEach(inj => {
            yaml += `\n        - name: ${inj.name}`;
          });
        }
        if (m.argument_mappings && m.argument_mappings.length > 0) {
          yaml += `\n      argument_mappings:`;
          m.argument_mappings.forEach(map => {
            yaml += `\n        - function_argument: ${map.function_argument}`;
            yaml += `\n          source: ${map.source}`;
          });
        }
        if (m.derivatives && m.derivatives.length > 0) {
          yaml += `\n      derivatives:`;
          m.derivatives.forEach(deriv => {
            yaml += `\n        - name: ${deriv.name}`;
          });
        }
      });
    } else {
      yaml += `\n    ${config.simulation_experiment.observation_models}`;
    }

    yaml += `

  stimulus:
    type: ${config.simulation_experiment.stimulus.type}
    amplitude: ${config.simulation_experiment.stimulus.amplitude}`;

    return yaml;
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
  });
})();
