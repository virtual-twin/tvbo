# Schema Improvements Proposal for tvbo_datamodel.yaml

Based on the comprehensive annotation of Schirner2023 using the tvbo-datamodel schema,
the following improvements are proposed for 100% adherence and better usability.

## CRITICAL: Schema vs Generated Code Mismatches

During validation, the following mismatches were found between the LinkML schema
definition and the generated Python datamodel:

### Issue 1: ProcessingStep `operation_type` vs `type`
The schema defines `operation_type` with `alias: type`, but the generated code
uses `type` as the field name. YAML files must use `type` not `operation_type`.

### Issue 2: ProcessingStep `function` vs `transformation`
The schema defines `function` with `alias: transformation`, but the generated code
uses `transformation` as the field name. YAML files must use `transformation`.

### Issue 3: Missing `description` slot in Coupling class
The `Coupling` class has slots `[name, label, parameters]` but NOT `description`.
Descriptions must be embedded in `label` field.

### Issue 4: Missing `description` slot in Parcellation class
The `Parcellation` class only has `[label]` as a slot, no `description`.

**Recommendation**: Either add `description` slot to these classes or regenerate
the Python datamodel to properly handle aliases.

---

## 1. SimulationStudy Top-Level Fields

### Issue
The `SimulationStudy` class is missing some commonly needed metadata fields that are
used in publications.

### Proposed Addition
```yaml
SimulationStudy:
  attributes:
    # ... existing attributes ...

    # NEW: Add abstract field for paper summary
    abstract:
      range: string
      description: "Brief abstract or summary of the simulation study."

    # NEW: Add authors field
    authors:
      multivalued: true
      range: string
      description: "List of authors for the publication."

    # NEW: Add keywords
    keywords:
      multivalued: true
      range: string
      description: "Keywords or tags for the study."
```

---

## 2. FittingTarget Enhancements

### Issue
The `FittingTarget` class lacks fields to specify:
- Data type (empirical vs simulated)
- Processing steps to derive the target
- Unit of the target metric

### Proposed Addition
```yaml
FittingTarget:
  slots:
    - label
    - equation
    - symbol
    - definition
    - parameters
    - unit  # ADD: unit slot
  attributes:
    # NEW: Specify the type of data
    target_type:
      range: string
      description: "Type of fitting target: 'correlation', 'distance', 'metric', etc."

    # NEW: Reference to empirical data
    empirical_data:
      range: string
      description: "Reference or path to empirical data used for fitting."

    # NEW: Reference to simulated data derivation
    simulated_from:
      range: Monitor
      description: "Monitor/observation model that produces the simulated data."
      inlined: false
```

---

## 3. CostFunction Enhancements

### Issue
The `CostFunction` class lacks:
- Definition field for human-readable explanation
- Reference to optimization method/algorithm
- Convergence criteria

### Proposed Addition
```yaml
CostFunction:
  slots:
    - label
    - equation
    - parameters
    - definition  # ADD: definition slot
  attributes:
    # NEW: Optimization algorithm used
    optimization_algorithm:
      range: string
      description: "Algorithm used for optimization: 'grid_search', 'differential_evolution', 'bayesian', etc."

    # NEW: Convergence criteria
    convergence_criterion:
      range: Equation
      description: "Criterion for optimization convergence."
      inlined: true

    # NEW: Number of iterations or evaluations
    max_iterations:
      range: integer
      description: "Maximum number of optimization iterations."
```

---

## 4. ModelFitting Enhancements

### Issue
The `ModelFitting` class should support:
- Multiple optimization stages
- Cross-validation specification
- Best-fit parameter storage

### Proposed Addition
```yaml
ModelFitting:
  slots:
    - label
    - description
  attributes:
    targets:
      multivalued: true
      range: FittingTarget
      inlined: true
    cost_function:
      range: CostFunction
      inlined: true

    # NEW: Optimization stage/phase
    stage:
      range: integer
      description: "Stage number in multi-stage optimization."

    # NEW: Free parameters being optimized
    free_parameters:
      multivalued: true
      range: Parameter
      inlined: true
      description: "Parameters being optimized in this fitting process."

    # NEW: Best-fit results
    best_fit_parameters:
      multivalued: true
      range: Parameter
      inlined: true
      description: "Optimized parameter values from fitting."

    # NEW: Goodness of fit metric achieved
    best_fit_metric:
      range: float
      description: "Best cost function value achieved."

    # NEW: Cross-validation specification
    cross_validation:
      range: CrossValidation
      inlined: true
      description: "Cross-validation strategy if applicable."
```

---

## 5. NEW: CrossValidation Class

### Proposed Addition
```yaml
CrossValidation:
  description: "Specification of cross-validation strategy for model fitting."
  slots:
    - label
    - description
  attributes:
    method:
      range: string
      description: "CV method: 'k-fold', 'leave-one-out', 'stratified', etc."
    n_folds:
      range: integer
      description: "Number of folds for k-fold CV."
    stratify_by:
      range: string
      description: "Variable to stratify by (e.g., 'group', 'subject')."
    test_size:
      range: float
      description: "Proportion of data for test set (for train/test splits)."
```

---

## 6. Parameter Free/Fitted Flag Enhancement

### Issue
The `Parameter` class has a `free` boolean but lacks:
- Clear distinction between fixed, free, and fitted states
- Storage of explored values from grid search

### Current Schema
```yaml
Parameter:
  attributes:
    free:
      range: boolean
    explored_values:
      multivalued: true
      range: float
      array: {}
```

### Proposed Enhancement
```yaml
Parameter:
  attributes:
    # Existing
    free:
      range: boolean
      description: "Whether this parameter is free for optimization."
    explored_values:
      multivalued: true
      range: float
      array: {}

    # NEW: More granular parameter status
    status:
      range: ParameterStatus
      description: "Current status: fixed, free, fitted, derived."

    # NEW: Fitted value (separate from default)
    fitted_value:
      range: float
      description: "Value determined by model fitting."

    # NEW: Confidence interval or uncertainty
    uncertainty:
      range: Range
      inlined: true
      description: "Uncertainty bounds on fitted parameter."

# NEW ENUM
enums:
  ParameterStatus:
    permissible_values:
      fixed:
        description: "Parameter is fixed at a specified value."
      free:
        description: "Parameter is free for optimization."
      fitted:
        description: "Parameter has been fitted to data."
      derived:
        description: "Parameter is derived from other parameters."
```

---

## 7. Tractogram Class Enhancement

### Issue
The `Tractogram` class exists but lacks:
- Integration with Network class (currently tractogram is just a string in Network)
- Subject-level vs group-level distinction

### Current Schema
```yaml
Network:
  attributes:
    tractogram:
      range: string  # Should be Tractogram class
```

### Proposed Change
```yaml
Network:
  attributes:
    tractogram:
      range: Tractogram  # CHANGE: Use Tractogram class instead of string
      inlined: true
      description: "Reference to tractography data used to derive connectivity."
```

---

## 8. Monitor Pipeline Enhancement

### Issue
The `Monitor` class inherits from `ObservationModel` which has a pipeline, but:
- The connection between `period` and pipeline `subsample` step is not clear
- Missing imaging-specific parameters (e.g., TR, TE for fMRI)

### Proposed Addition
```yaml
Monitor:
  is_a: ObservationModel
  attributes:
    # Existing
    period:
      range: float
      description: "Sampling period for the monitor"
    imaging_modality:
      range: ImagingModality

    # NEW: Repetition time (for fMRI)
    repetition_time:
      range: float
      alias: TR
      description: "fMRI repetition time in ms."
      unit: "ms"

    # NEW: Link to hemodynamic model parameters
    hemodynamic_model:
      range: string
      description: "Hemodynamic model used: 'balloon_windkessel', 'canonical_hrf', etc."

    # NEW: Explicit output variables selection
    output_variables:
      multivalued: true
      range: string
      description: "Which state variables to observe."
```

---

## 9. Sample Class Enhancement

### Issue
The `Sample` class is minimal and lacks fields for:
- Demographics
- Clinical data references
- Data source/dataset

### Proposed Enhancement
```yaml
Sample:
  attributes:
    groups:
      multivalued: true
      range: string
    size:
      range: integer

    # NEW: Group sizes
    group_sizes:
      multivalued: true
      range: integer
      description: "Size of each group in order."

    # NEW: Demographics summary
    demographics:
      range: string
      description: "Summary of sample demographics."

    # NEW: Dataset reference
    dataset:
      range: string
      description: "Source dataset (e.g., 'ADNI', 'HCP', 'UK Biobank')."

    # NEW: Inclusion/exclusion criteria
    inclusion_criteria:
      multivalued: true
      range: string

    exclusion_criteria:
      multivalued: true
      range: string
```

---

## 10. Network Weights/Lengths Matrix Enhancement

### Issue
The `Network` class has `Matrix` for weights/lengths in the old Connectome-style
representation, but the current schema doesn't have explicit `weights` and `lengths`
attributes.

### Analysis
Looking at the schema, the Network class uses:
- `nodes` and `edges` for explicit representation
- But lacks direct `weights` and `lengths` matrix attributes for the traditional
  connectome representation used in TVB.

### Proposed Addition
```yaml
Network:
  attributes:
    # Existing node/edge representation...

    # NEW: Matrix-based representation (for backward compatibility)
    weights:
      range: Matrix
      inlined: true
      description: "Structural connectivity weights matrix (alternative to edges)."

    lengths:
      range: Matrix
      inlined: true
      description: "Fiber tract lengths matrix (for computing delays)."

    region_labels:
      multivalued: true
      range: string
      description: "Labels for each region/node in matrix representation."
```

---

## Summary of Proposed Changes

| Priority | Class/Enum | Change Type | Description |
|----------|------------|-------------|-------------|
| High | ModelFitting | Enhancement | Add free_parameters, best_fit_parameters, best_fit_metric |
| High | FittingTarget | Enhancement | Add target_type, empirical_data, unit |
| High | CostFunction | Enhancement | Add optimization_algorithm, definition |
| High | Network | Fix | Change tractogram from string to Tractogram |
| Medium | Parameter | Enhancement | Add status enum, fitted_value, uncertainty |
| Medium | Monitor | Enhancement | Add repetition_time, hemodynamic_model |
| Medium | Sample | Enhancement | Add group_sizes, dataset, criteria |
| Medium | SimulationStudy | Enhancement | Add abstract, authors, keywords |
| Low | Network | Enhancement | Add weights, lengths matrices for backward compat |
| Low | NEW | Class | Add CrossValidation class |

---

## Implementation Notes

1. All proposed changes are backward-compatible (adding optional fields)
2. Existing YAML files will continue to validate
3. The changes align with common neuroimaging/computational modeling metadata standards
4. Consider BIDS-Model extension alignment for fitting metadata

---

## Validation

After implementing these changes, the Schirner2023.yaml file should validate with
full coverage of all concepts from the publication without workarounds.
