/**
 * TVBO Icon & Gradient Registry
 *
 * Shared icon/gradient definitions for use across the docs.
 * Keys are page slugs (filename without extension).
 *
 * Usage from any page:
 *   <script src="/_static/tvbo-icons.js"></script>
 *   <script>
 *     const info = TVBO_ICONS["BifurcationAnalysis"];
 *     // info.icon     → "fa-solid fa-code-branch"
 *     // info.gradient → "linear-gradient(135deg, #c471f5 0%, #fa71cd 100%)"
 *     // info.color    → "#c471f5"   (primary color for badges, borders, etc.)
 *   </script>
 *
 * Helper:  TVBO_ICONS.badge(slug)
 *   Returns HTML string for a small icon badge with the gradient background.
 */

/* global window */
window.TVBO_ICONS = {

  // ── Fundamentals ──
  DefineDynamicalSystem: {
    icon: "fa-solid fa-wave-square",
    gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "#667eea",
  },
  DefineNetworks: {
    icon: "fa-solid fa-diagram-project",
    gradient: "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    color: "#43e97b",
  },
  Network: {
    icon: "fa-solid fa-diagram-project",
    gradient: "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    color: "#43e97b",
  },
  SimulationExperiments: {
    icon: "fa-solid fa-flask",
    gradient: "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
    color: "#fa709a",
  },

  // ── Code Generation ──
  CodeGeneration: {
    icon: "fa-solid fa-code",
    gradient: "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    color: "#a18cd1",
  },

  // ── Observation & Analysis ──
  ObservationModels: {
    icon: "fa-solid fa-eye",
    gradient: "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)",
    color: "#fcb69f",
  },
  ParameterExploration: {
    icon: "fa-solid fa-magnifying-glass-chart",
    gradient: "linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%)",
    color: "#66a6ff",
  },
  ModelFitting: {
    icon: "fa-solid fa-chart-line",
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    color: "#f5576c",
  },
  LossFunction: {
    icon: "fa-solid fa-chart-line",
    gradient: "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    color: "#f5576c",
  },
  Algorithms: {
    icon: "fa-solid fa-gears",
    gradient: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    color: "#4facfe",
  },
  BifurcationAnalysis: {
    icon: "fa-solid fa-code-branch",
    gradient: "linear-gradient(135deg, #c471f5 0%, #fa71cd 100%)",
    color: "#c471f5",
  },

  // ── Simulation Features ──
  Events: {
    icon: "fa-solid fa-bolt",
    gradient: "linear-gradient(135deg, #f6d365 0%, #fda085 100%)",
    color: "#f6d365",
  },
  Noise: {
    icon: "fa-solid fa-shuffle",
    gradient: "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)",
    color: "#a1c4fd",
  },

  // ── Advanced ──
  PDESimulation: {
    icon: "fa-solid fa-border-all",
    gradient: "linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)",
    color: "#96e6a1",
  },
  HeterogeneousNodes: {
    icon: "fa-solid fa-circle-nodes",
    gradient: "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)",
    color: "#84fab0",
  },
  HeterogeneousEdges: {
    icon: "fa-solid fa-arrow-right-arrow-left",
    gradient: "linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)",
    color: "#fbc2eb",
  },
  UnitValidation: {
    icon: "fa-solid fa-ruler-combined",
    gradient: "linear-gradient(135deg, #fddb92 0%, #d1fdff 100%)",
    color: "#fddb92",
  },

  // ── Specification ──
  Output: {
    icon: "fa-solid fa-file-export",
    gradient: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    color: "#667eea",
  },

  // ── Interoperability ──
  BIDS: {
    icon: "fa-solid fa-database",
    gradient: "linear-gradient(135deg, #f7971e 0%, #ffd200 100%)",
    color: "#f7971e",
  },
  OpenMINDS: {
    icon: "fa-solid fa-brain",
    gradient: "linear-gradient(135deg, #11998e 0%, #38ef7d 100%)",
    color: "#11998e",
  },
  PyRates: {
    icon: "fa-solid fa-chart-area",
    gradient: "linear-gradient(135deg, #ee9ca7 0%, #ffdde1 100%)",
    color: "#ee9ca7",
  },
  tvboptim: {
    icon: "fa-solid fa-sliders",
    gradient: "linear-gradient(135deg, #fc5c7d 0%, #6a82fb 100%)",
    color: "#6a82fb",
  },
  NetworkDynamics: {
    icon: "fa-solid fa-circle-nodes",
    gradient: "linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)",
    color: "#84fab0",
  },
  BifurcationKit: {
    icon: "fa-solid fa-code-branch",
    gradient: "linear-gradient(135deg, #c471f5 0%, #fa71cd 100%)",
    color: "#c471f5",
  },
  ModelingToolkit: {
    icon: "fa-solid fa-gears",
    gradient: "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    color: "#4facfe",
  },

  // ── Helpers ──

  /** Return HTML for a small circular icon badge. */
  badge(slug) {
    const info = this[slug];
    if (!info) return "";
    return `<span style="display:inline-flex;align-items:center;justify-content:center;`
      + `width:1.6em;height:1.6em;border-radius:50%;background:${info.gradient};`
      + `color:#fff;font-size:0.85em;vertical-align:middle;margin-right:0.3em;">`
      + `<i class="${info.icon}" aria-hidden="true"></i></span>`;
  },

  /** Return an icon <i> tag for a slug. */
  iconTag(slug) {
    const info = this[slug];
    return info ? `<i class="${info.icon}" aria-hidden="true"></i>` : "";
  },

  /** Create a colored placeholder div (gradient + large icon). */
  placeholder(slug) {
    const info = this[slug];
    if (!info) return document.createElement("div");
    const div = document.createElement("div");
    div.style.background = info.gradient;
    div.style.display = "flex";
    div.style.alignItems = "center";
    div.style.justifyContent = "center";
    div.style.width = "100%";
    div.style.height = "100%";
    div.style.color = "#fff";
    div.style.fontSize = "2.2rem";
    div.innerHTML = `<i class="${info.icon}" aria-hidden="true"></i>`;
    return div;
  },
};
