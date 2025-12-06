/**
 * 3D Network Graph Visualization
 * Uses Three.js to render nodes and edges with optional brain mesh overlay
 * All coordinates are in MNI space (no transformation needed)
 */

(function() {
  'use strict';

  console.log('[NetworkGraph3D] Script loaded and executing');

  let scene, camera, renderer, controls;
  let brainMesh = null;
  let brainMeshVisible = true;
  let nodeMeshes = [];
  let edgeLines = [];
  let selectedNode = null;
  let raycaster, mouse;

  const NODE_RADIUS = 5;
  const NODE_COLOR = 0xff6b6b;
  const NODE_SELECTED_COLOR = 0x4ecdc4;
  const EDGE_COLOR = 0x95a5a6;
  const BRAIN_COLOR = 0x667788;
  const BRAIN_OPACITY = 0.15;

  /**
   * Initialize the 3D scene
   */
  function initScene(containerId) {
    console.log('[NetworkGraph3D] initScene called with containerId:', containerId);
    const container = document.getElementById(containerId);
    if (!container) {
      console.error('[NetworkGraph3D] Container not found:', containerId);
      return null;
    }
    console.log('[NetworkGraph3D] Container found:', container);

    const width = container.clientWidth;
    const height = container.clientHeight;
    console.log('[NetworkGraph3D] Container dimensions:', width, 'x', height);

    // Check if THREE is available
    if (typeof THREE === 'undefined') {
      console.error('[NetworkGraph3D] THREE.js is not loaded!');
      return null;
    }
    console.log('[NetworkGraph3D] THREE.js version:', THREE.REVISION);

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    console.log('[NetworkGraph3D] Scene created');

    // Camera - positioned for sagittal view (from the right side)
    // MNI space: X=left-right, Y=posterior-anterior, Z=inferior-superior
    // For sagittal view: camera on +X axis, looking at brain center, Z is UP
    const brainCenterY = -17; // Approximate center: (min -107 + max +73) / 2
    const brainCenterZ = 5;   // Approximate center: (min -72 + max +82) / 2
    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 2000);
    camera.position.set(250, brainCenterY, brainCenterZ); // Right side view (sagittal)
    camera.up.set(0, 0, 1); // Z is up in MNI space (superior direction)
    camera.lookAt(0, brainCenterY, brainCenterZ);
    console.log('[NetworkGraph3D] Camera created at (250, -17, 5) - sagittal view, Z-up');

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    console.log('[NetworkGraph3D] Renderer created and added to container');

    // Controls - check if OrbitControls exists
    if (typeof THREE.OrbitControls === 'undefined') {
      console.error('[NetworkGraph3D] THREE.OrbitControls is not available! Make sure OrbitControls.js is loaded.');
      // Try to continue without controls
      controls = null;
    } else {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.target.set(0, brainCenterY, brainCenterZ); // Look at brain center
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.minDistance = 50;
      controls.maxDistance = 500;
      controls.update();
      console.log('[NetworkGraph3D] OrbitControls created, target at brain center');
    }

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(100, 100, 100);
    scene.add(directionalLight);
    console.log('[NetworkGraph3D] Lights added');

    // Raycaster for node selection
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Event listeners
    renderer.domElement.addEventListener('click', onMouseClick, false);
    window.addEventListener('resize', () => onWindowResize(container), false);

    // Start animation
    console.log('[NetworkGraph3D] Starting animation loop');
    animate();

    console.log('[NetworkGraph3D] initScene completed successfully');
    return { scene, camera, renderer, controls };
  }

  let animationFrameCount = 0;
  function animate() {
    requestAnimationFrame(animate);
    if (controls) {
      controls.update();
    }
    renderer.render(scene, camera);
    // Log every 300 frames (roughly every 5 seconds at 60fps)
    animationFrameCount++;
    if (animationFrameCount === 1) {
      console.log('[NetworkGraph3D] First animation frame rendered');
    }
  }

  function onWindowResize(container) {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function onMouseClick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(nodeMeshes);

    if (intersects.length > 0) {
      const clickedNode = intersects[0].object;
      selectNode(clickedNode);
    }
  }

  function selectNode(nodeMesh) {
    // Deselect previous
    if (selectedNode) {
      selectedNode.material.color.setHex(NODE_COLOR);
      selectedNode.material.emissive.setHex(0x000000);
    }

    // Select new
    selectedNode = nodeMesh;
    selectedNode.material.color.setHex(NODE_SELECTED_COLOR);
    selectedNode.material.emissive.setHex(NODE_SELECTED_COLOR);
    selectedNode.material.emissiveIntensity = 0.3;

    // Highlight corresponding input row
    const nodeId = nodeMesh.userData.nodeId;
    highlightNodeRow(nodeId);

    // Show edit popup
    showNodeEditor(nodeMesh);
  }

  function highlightNodeRow(nodeId) {
    const rows = document.querySelectorAll('#customNetworkNodes .builder-row');
    rows.forEach(row => {
      if (row.dataset.nodeId == nodeId) {
        row.style.boxShadow = '0 0 0 2px #4ecdc4';
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        row.style.boxShadow = '';
      }
    });
  }

  function showNodeEditor(nodeMesh) {
    const nodeId = nodeMesh.userData.nodeId;
    const row = document.querySelector(`#customNetworkNodes .builder-row[data-node-id="${nodeId}"]`);
    if (row) {
      // Focus the label input
      const labelInput = row.querySelector('.node-label');
      if (labelInput) {
        labelInput.focus();
        labelInput.select();
      }
    }
  }

  /**
   * Create a node sphere at MNI coordinates (used directly, no transformation)
   */
  function createNodeMesh(id, x, y, z, label) {
    console.log('[NetworkGraph3D] createNodeMesh MNI coords:', { id, x, y, z, label });

    const geometry = new THREE.SphereGeometry(NODE_RADIUS, 16, 16);
    const material = new THREE.MeshPhongMaterial({
      color: NODE_COLOR,
      emissive: 0x000000,
      shininess: 100
    });
    const sphere = new THREE.Mesh(geometry, material);
    // Use MNI coordinates directly (X=left-right, Y=posterior-anterior, Z=inferior-superior)
    sphere.position.set(x || 0, y || 0, z || 0);
    sphere.userData = { nodeId: id, label: label };
    return sphere;
  }

  /**
   * Create an edge as a cylinder between two nodes (thicker than lines)
   */
  function createEdgeLine(node1, node2, weight = 1) {
    console.log('[NetworkGraph3D] createEdgeLine between:', node1?.userData?.nodeId, 'and', node2?.userData?.nodeId);

    // Calculate edge properties
    const start = node1.position;
    const end = node2.position;
    const direction = new THREE.Vector3().subVectors(end, start);
    const length = direction.length();
    const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);

    // Edge thickness based on weight (min 0.5, max 2)
    const radius = 0.3 + (weight * 0.5);

    // Create cylinder geometry
    const geometry = new THREE.CylinderGeometry(radius, radius, length, 8, 1);
    const material = new THREE.MeshPhongMaterial({
      color: EDGE_COLOR,
      transparent: true,
      opacity: 0.7
    });

    const cylinder = new THREE.Mesh(geometry, material);

    // Position at midpoint
    cylinder.position.copy(midpoint);

    // Orient cylinder to point from start to end
    cylinder.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), // Cylinder default axis
      direction.normalize()
    );

    return cylinder;
  }

  /**
   * Update the graph from the node/edge form inputs
   */
  function updateGraphFromInputs() {
    console.log('[NetworkGraph3D] updateGraphFromInputs called');
    if (!scene) {
      console.warn('[NetworkGraph3D] Scene not initialized, skipping update');
      return;
    }

    // Clear existing
    console.log('[NetworkGraph3D] Clearing', nodeMeshes.length, 'nodes and', edgeLines.length, 'edges');
    nodeMeshes.forEach(m => scene.remove(m));
    edgeLines.forEach(l => scene.remove(l));
    nodeMeshes = [];
    edgeLines = [];

    // Get nodes from inputs
    const nodeRows = document.querySelectorAll('#customNetworkNodes .builder-row');
    console.log('[NetworkGraph3D] Found', nodeRows.length, 'node rows');
    const nodeMap = {};

    nodeRows.forEach(row => {
      const id = row.dataset.nodeId;
      const label = row.querySelector('.node-label')?.value || `Node ${id}`;
      const x = parseFloat(row.querySelector('.node-x')?.value) || 0;
      const y = parseFloat(row.querySelector('.node-y')?.value) || 0;
      const z = parseFloat(row.querySelector('.node-z')?.value) || 0;
      console.log('[NetworkGraph3D] Processing node row:', { id, label, x, y, z });

      const nodeMesh = createNodeMesh(id, x, y, z, label);
      scene.add(nodeMesh);
      nodeMeshes.push(nodeMesh);
      nodeMap[id] = nodeMesh;
    });

    // Get edges from inputs
    const edgeRows = document.querySelectorAll('#customNetworkEdges .builder-row');
    console.log('[NetworkGraph3D] Found', edgeRows.length, 'edge rows');
    edgeRows.forEach(row => {
      const sourceId = row.querySelector('.edge-source-select')?.value;
      const targetId = row.querySelector('.edge-target-select')?.value;
      const weight = parseFloat(row.querySelector('.edge-weight')?.value) || 1;

      if (sourceId && targetId && nodeMap[sourceId] && nodeMap[targetId]) {
        const edgeLine = createEdgeLine(nodeMap[sourceId], nodeMap[targetId], weight);
        scene.add(edgeLine);
        edgeLines.push(edgeLine);
      }
    });

    // Auto-fit camera to show brain extent (always, even with 0 nodes)
    fitCameraToNodes();

    // Update node/edge count display in persistent panel
    const nodeCountEl = document.getElementById('networkViewerNodeCount');
    const edgeCountEl = document.getElementById('networkViewerEdgeCount');
    if (nodeCountEl) nodeCountEl.textContent = `Nodes: ${nodeMeshes.length}`;
    if (edgeCountEl) edgeCountEl.textContent = `Edges: ${edgeLines.length}`;
  }

  /**
   * Fit camera to show full brain mesh extent (with padding)
   * If brain mesh is disabled, zooms to fit nodes only
   */
  function fitCameraToNodes() {
    let box;

    if (brainMeshVisible && brainMesh) {
      // Brain mesh is visible - show full brain extent
      // MNI brain extent (approximate bounding box)
      // X: -80 to +80 (left-right)
      // Y: -110 to +80 (posterior-anterior)
      // Z: -60 to +90 (inferior-superior)
      const brainMin = new THREE.Vector3(-80, -110, -60);
      const brainMax = new THREE.Vector3(80, 80, 90);
      box = new THREE.Box3(brainMin, brainMax);

      // Expand to include all nodes if they exist
      if (nodeMeshes.length > 0) {
        nodeMeshes.forEach(m => box.expandByObject(m));
      }
    } else {
      // Brain mesh disabled - zoom to nodes only
      if (nodeMeshes.length === 0) {
        // No nodes and no brain mesh - use default brain extent
        const brainMin = new THREE.Vector3(-80, -110, -60);
        const brainMax = new THREE.Vector3(80, 80, 90);
        box = new THREE.Box3(brainMin, brainMax);
      } else {
        box = new THREE.Box3();
        nodeMeshes.forEach(m => box.expandByObject(m));
        // Add some padding around nodes
        box.expandByScalar(20);
      }
    }

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    // Position camera with padding (1.2x the max dimension)
    const distance = maxDim * 1.2;

    // Sagittal view: camera on +X axis, looking at center
    camera.position.set(center.x + distance, center.y, center.z);
    camera.up.set(0, 0, 1); // Z is up in MNI space

    if (controls) {
      controls.target.copy(center);
      controls.update();
    }

    console.log('[NetworkGraph3D] Camera fit, brainMeshVisible:', brainMeshVisible, 'distance:', distance);
  }

  /**
   * Load and parse GIFTI brain mesh
   */
  async function loadBrainMesh(url) {
    try {
      const response = await fetch(url);
      const text = await response.text();

      // Parse GIFTI XML
      const parser = new DOMParser();
      const xml = parser.parseFromString(text, 'text/xml');

      const dataArrays = xml.querySelectorAll('DataArray');
      let vertices = null;
      let faces = null;

      for (const da of dataArrays) {
        const intent = da.getAttribute('Intent');
        const dataText = da.querySelector('Data')?.textContent?.trim();

        if (!dataText) continue;

        // Decode base64 gzipped data
        const binaryString = atob(dataText);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }

        // Decompress with pako if available, otherwise skip
        let decompressed;
        if (typeof pako !== 'undefined') {
          try {
            decompressed = pako.inflate(bytes);
          } catch (e) {
            console.warn('Failed to decompress GIFTI data:', e);
            continue;
          }
        } else {
          console.warn('pako not available for GIFTI decompression');
          continue;
        }

        const dim0 = parseInt(da.getAttribute('Dim0'));
        const dim1 = parseInt(da.getAttribute('Dim1')) || 1;
        const dataType = da.getAttribute('DataType');

        if (intent === 'NIFTI_INTENT_POINTSET') {
          // Vertices (float32)
          vertices = new Float32Array(decompressed.buffer);
        } else if (intent === 'NIFTI_INTENT_TRIANGLE') {
          // Faces (int32)
          faces = new Int32Array(decompressed.buffer);
        }
      }

      if (vertices && faces) {
        createBrainGeometry(vertices, faces);
      }
    } catch (error) {
      console.error('Error loading brain mesh:', error);
    }
  }

  /**
   * Create Three.js geometry from vertices and faces
   */
  function createBrainGeometry(vertices, faces) {
    const geometry = new THREE.BufferGeometry();

    // Set vertices
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));

    // Set faces
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));

    // Compute normals
    geometry.computeVertexNormals();

    // Create mesh
    const material = new THREE.MeshPhongMaterial({
      color: BRAIN_COLOR,
      transparent: true,
      opacity: BRAIN_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false
    });

    brainMesh = new THREE.Mesh(geometry, material);
    brainMesh.visible = brainMeshVisible;
    scene.add(brainMesh);
  }

  /**
   * Load brain mesh from OBJ file - NO TRANSFORMATION, keep in MNI space
   */
  function loadBrainMeshFromOBJ() {
    console.log('[NetworkGraph3D] loadBrainMeshFromOBJ called');

    // Check if OBJLoader is available
    if (typeof THREE.OBJLoader === 'undefined') {
      console.error('[NetworkGraph3D] THREE.OBJLoader is not available! Creating placeholder instead.');
      createPlaceholderBrain();
      return;
    }

    const loader = new THREE.OBJLoader();
    const objPath = '/tvbo/static/src/img/mni152_2009.obj';
    console.log('[NetworkGraph3D] Loading OBJ from:', objPath);

    loader.load(
      objPath,
      function(object) {
        console.log('[NetworkGraph3D] OBJ loaded successfully:', object);

        // Apply material to all meshes in the loaded object
        object.traverse(function(child) {
          if (child instanceof THREE.Mesh) {
            child.material = new THREE.MeshPhongMaterial({
              color: BRAIN_COLOR,
              transparent: true,
              opacity: BRAIN_OPACITY,
              side: THREE.DoubleSide,
              depthWrite: false
            });
          }
        });

        // Log the mesh bounds (for debugging)
        const box = new THREE.Box3().setFromObject(object);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        console.log('[NetworkGraph3D] Brain mesh in MNI space - size:', size, 'center:', center);

        // NO TRANSFORMATION - keep mesh in original MNI coordinates
        // MNI space: X=left-right (-72 to +72), Y=posterior-anterior (-107 to +73), Z=inferior-superior (-72 to +82)

        brainMesh = object;
        brainMesh.visible = brainMeshVisible;
        scene.add(brainMesh);
        console.log('[NetworkGraph3D] Brain mesh added to scene (MNI space, no transform)');

        // Update graph
        setTimeout(updateGraphFromInputs, 100);
      },
      function(xhr) {
        console.log('[NetworkGraph3D] OBJ loading progress:', (xhr.loaded / xhr.total * 100).toFixed(1) + '%');
      },
      function(error) {
        console.error('[NetworkGraph3D] Error loading OBJ:', error);
        console.log('[NetworkGraph3D] Falling back to placeholder brain');
        createPlaceholderBrain();
      }
    );
  }

  /**
   * Create a simple placeholder brain outline (fallback)
   */
  function createPlaceholderBrain() {
    console.log('[NetworkGraph3D] createPlaceholderBrain called');
    // Create a simple ellipsoid as brain placeholder
    const geometry = new THREE.SphereGeometry(70, 32, 24);
    geometry.scale(1, 0.8, 1.2); // Make it brain-shaped

    const material = new THREE.MeshPhongMaterial({
      color: BRAIN_COLOR,
      transparent: true,
      opacity: BRAIN_OPACITY,
      side: THREE.DoubleSide,
      depthWrite: false,
      wireframe: false
    });

    brainMesh = new THREE.Mesh(geometry, material);
    brainMesh.visible = brainMeshVisible;
    scene.add(brainMesh);
    console.log('[NetworkGraph3D] Placeholder brain added to scene');
  }

  /**
   * Toggle brain mesh visibility
   */
  function toggleBrainMesh() {
    console.log('[NetworkGraph3D] toggleBrainMesh called, current visibility:', brainMeshVisible);
    if (brainMesh) {
      brainMeshVisible = !brainMeshVisible;
      brainMesh.visible = brainMeshVisible;
      console.log('[NetworkGraph3D] Brain mesh visibility set to:', brainMeshVisible);
      // Adjust camera based on new visibility
      fitCameraToNodes();
    } else {
      console.warn('[NetworkGraph3D] Brain mesh not available');
    }
  }

  /**
   * Reset camera to default position showing full brain extent
   */
  function resetCamera() {
    console.log('[NetworkGraph3D] resetCamera called');
    // Use the same brain-extent fitting as fitCameraToNodes
    fitCameraToNodes();
  }

  /**
   * Initialize the network graph visualization
   */
  function initNetworkGraph() {
    console.log('[NetworkGraph3D] initNetworkGraph called');
    // Use persistent container (always visible in right panel)
    const containerId = 'persistentNetworkGraph3D';
    const container = document.getElementById(containerId);
    if (!container) {
      // Fallback to old container for backwards compatibility
      const oldContainer = document.getElementById('networkGraph3D');
      if (oldContainer) {
        console.log('[NetworkGraph3D] Using fallback container #networkGraph3D');
        return initWithContainer('networkGraph3D');
      }
      console.warn('[NetworkGraph3D] No container found, will retry...');
      return;
    }
    console.log('[NetworkGraph3D] Container found, initializing scene...');
    return initWithContainer(containerId);
  }

  function initWithContainer(containerId) {
    // Initialize scene
    const result = initScene(containerId);
    if (!result) {
      console.error('[NetworkGraph3D] Failed to initialize scene');
      return;
    }

    // Load brain mesh from OBJ file
    console.log('[NetworkGraph3D] Loading brain mesh from OBJ...');
    loadBrainMeshFromOBJ();

    // Add axis helper for orientation (at origin)
    const axesHelper = new THREE.AxesHelper(80);
    scene.add(axesHelper);
    console.log('[NetworkGraph3D] Added axes helper');

    // Wire up controls - try persistent buttons first, then fallback
    const toggleBtn = document.getElementById('persistentToggleBrainMesh') || document.getElementById('toggleBrainMesh');
    const resetBtn = document.getElementById('persistentResetCamera') || document.getElementById('resetCamera');
    console.log('[NetworkGraph3D] Toggle brain button:', toggleBtn ? 'found' : 'not found');
    console.log('[NetworkGraph3D] Reset camera button:', resetBtn ? 'found' : 'not found');

    toggleBtn?.addEventListener('click', toggleBrainMesh);
    resetBtn?.addEventListener('click', resetCamera);

    // Listen for node/edge changes
    const nodesContainer = document.getElementById('customNetworkNodes');
    const edgesContainer = document.getElementById('customNetworkEdges');
    console.log('[NetworkGraph3D] Nodes container:', nodesContainer ? 'found' : 'not found');
    console.log('[NetworkGraph3D] Edges container:', edgesContainer ? 'found' : 'not found');

    if (nodesContainer) {
      const observer = new MutationObserver(() => {
        console.log('[NetworkGraph3D] Nodes container mutation detected');
        setTimeout(updateGraphFromInputs, 100);
      });
      observer.observe(nodesContainer, { childList: true, subtree: true, attributes: true });
      console.log('[NetworkGraph3D] MutationObserver attached to nodes container');
    }

    if (edgesContainer) {
      const observer = new MutationObserver(() => {
        console.log('[NetworkGraph3D] Edges container mutation detected');
        setTimeout(updateGraphFromInputs, 100);
      });
      observer.observe(edgesContainer, { childList: true, subtree: true, attributes: true });
      console.log('[NetworkGraph3D] MutationObserver attached to edges container');
    }

    // Initial update
    console.log('[NetworkGraph3D] Scheduling initial graph update...');
    setTimeout(updateGraphFromInputs, 500);
    console.log('[NetworkGraph3D] initNetworkGraph completed');
  }

  // Flag to prevent double initialization
  let initialized = false;

  function safeInit() {
    if (initialized) {
      console.log('[NetworkGraph3D] Already initialized, skipping');
      return;
    }
    // Check persistent container first, then fallback
    let container = document.getElementById('persistentNetworkGraph3D');
    if (!container) {
      container = document.getElementById('networkGraph3D');
    }
    if (!container) {
      console.log('[NetworkGraph3D] Container not found yet');
      return;
    }
    // Check if container is visible (has dimensions)
    if (container.offsetWidth === 0 || container.offsetHeight === 0) {
      console.log('[NetworkGraph3D] Container has no dimensions (probably hidden)');
      return;
    }
    initialized = true;
    initNetworkGraph();
  }

  // Export for use
  window.NetworkGraph3D = {
    init: safeInit,
    update: updateGraphFromInputs,
    toggleBrain: toggleBrainMesh,
    resetCamera: resetCamera,
    loadBrainMesh: loadBrainMesh
  };
  console.log('[NetworkGraph3D] Exported to window.NetworkGraph3D');

  // Try to init on DOMContentLoaded, but also allow manual init from tab switch
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      console.log('[NetworkGraph3D] DOMContentLoaded fired');
      setTimeout(safeInit, 500);
    });
  } else {
    // DOM already loaded
    console.log('[NetworkGraph3D] DOM already loaded, trying init...');
    setTimeout(safeInit, 500);
  }

  console.log('[NetworkGraph3D] Script initialization complete');

})();
