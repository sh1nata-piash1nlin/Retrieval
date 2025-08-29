// Icons for each search type
const ICONS = {
  "InternVideo2 Search": "fa-solid fa-video",
  "blip2 Search": "fa-solid fa-robot",
  "SigLIP2 Search": "fa-solid fa-image",
  "bge-m3 Search": "fa-solid fa-brain",
  "FDP Search": "fa-solid fa-database",
  "PE Search": "fa-solid fa-microchip",
  "Fusion Search": "fa-regular fa-circle",
  "Local Search": "fa-regular fa-gem",
  "Group Search": "fa-solid fa-users",
  "OCR Match": "fa-regular fa-star",
  "Subtitle Match": "fa-solid fa-align-left",
  "Similar Image Search": "fa-regular fa-image",
  "Similar Frame Search": "fa-solid fa-image"
};

// HTML templates for each tab (replace with your real markup / forms)
const TEMPLATES = {
  "InternVideo2 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-success" id="internvideo2-search-btn">Search</button>
        <button class="btn btn-danger" id="internvideo2-submit-btn">Submit</button>
      </div>
      <div></div>
    </div>
    <div class="search-results"></div>
  `,
  "blip2 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-primary" id="blip2-search-btn">Search</button>
        <button class="btn btn-danger" id="blip2-submit-btn">Submit</button>
      </div>
      <div></div>
    </div>
    <div class="search-results"></div>
  `,
  "SigLIP2 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm SigLIP2..." />
        <button class="btn btn-warning" id="siglip2-search-btn">Search</button>
        <button class="btn btn-danger" id="siglip2-submit-btn">Submit</button>
      </div>
      <div></div>
    </div>
    <div class="search-results"></div>
  `,
  "bge-m3 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-primary" id="bge-m3-search-btn">Search</button>
        <button class="btn btn-danger" id="bge-m3-submit-btn">Submit</button>
      </div>
      <div></div>
    </div>
    <div class="search-results"></div>
  `,
  "PE Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm PE..." />
        <button class="btn btn-warning" id="pe-search-btn">Search</button>
        <button class="btn btn-danger" id="pe-submit-btn">Submit</button>
      </div>
      <div></div>
    </div>
    <div class="search-results"></div>
  `,
  "FDP Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm FDP..." />
        <button class="btn btn-info" id="fdp-search-btn">Search</button>
        <button class="btn btn-danger" id="fdp-submit-btn">Submit</button>
      </div>
      <div></div>
    </div>
    <div class="search-results"></div>
  `,
  "OCR Match": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="text" class="form-control" placeholder="Nhập nội dung OCR cần tìm…" />
      <button class="btn btn-primary">Search OCR</button>
      <button class="btn btn-danger" id="ocr-submit-btn">Submit</button>
    </div>
  `,
  "Subtitle Match": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="text" class="form-control" placeholder="Nhập câu phụ đề cần tìm…" />
      <button class="btn btn-primary">Search Subtitle</button>
      <button class="btn btn-danger" id="subtitle-submit-btn">Submit</button>
    </div>
  `,
  "Similar Image Search": () => `
    <input type="file" class="form-control mb-3" accept="image/*" />
    <button class="btn btn-primary">Find Similar Images</button>
    <button class="btn btn-danger" id="similar-image-submit-btn">Submit</button>
  `
};

const tabBar = document.getElementById('tabBar');
const tabContent = document.getElementById('tabContent');
const dropdownMenu = document.getElementById('searchMenu');
const dropdownBtnSpan = document.getElementById('searchDropdown').querySelector('span');

// Open a tab (or activate if existing)
function openTab(label) {
  const id = label.toLowerCase().replace(/\s+/g, '-');
  const existingLink = document.getElementById(`tab-${id}-link`);
  if (existingLink) {
    new bootstrap.Tab(existingLink).show();
    toggleSidebarControls(label);
    return;
  }

  // Create tab header
  const li = document.createElement('li');
  li.className = 'nav-item';
  li.innerHTML = `
    <a class="nav-link" id="tab-${id}-link" data-bs-toggle="tab" href="#tab-${id}" role="tab" aria-controls="tab-${id}" aria-selected="false">
      <i class="${ICONS[label]} me-1"></i> ${label}
      <button type="button" class="btn-close" aria-label="Close" data-close="${id}"></button>
    </a>`;
  tabBar.appendChild(li);

  // Create tab pane
  const pane = document.createElement('div');
  pane.className = 'tab-pane fade';
  pane.id = `tab-${id}`;
  pane.setAttribute('role', 'tabpanel');
  pane.innerHTML = (TEMPLATES[label] ? TEMPLATES[label]() : `<p>${label} content…</p>`);
  tabContent.appendChild(pane);

  // Show new tab
  new bootstrap.Tab(li.querySelector('a')).show();
  toggleSidebarControls(label);
}

function toggleSidebarControls(label) {
  const kRow = document.getElementById('kRowContainer');
  const hierarchyKRow = document.getElementById('hierarchyKRowContainer');
  const groupImagesContainer = document.getElementById('groupImagesPerVideoContainer');

  // Hide all controls first
  kRow.style.display = 'none';
  hierarchyKRow.style.display = 'none';
  groupImagesContainer.style.display = 'none';

  // Show relevant controls based on search type
  switch (label) {
    case 'Single Search':
      kRow.style.display = 'block'; break;
    case 'Hierarchy Search':
      hierarchyKRow.style.display = 'block'; break;
    case 'Similar Image Search':
      kRow.style.display = 'block';
      groupImagesContainer.style.display = 'block';
      break;
    case 'Subtitle Match':
      kRow.style.display = 'block'; break;
    default:
      kRow.style.display = 'block';
  }
}

function setDropdownLabel(label){
  dropdownBtnSpan.innerHTML = `<i class="${ICONS[label]}"></i> ${label}`;
}

// Handle dropdown click
dropdownMenu.addEventListener('click', (e) => {
  const item = e.target.closest('.dropdown-item-custom');
  if (!item) return;
  e.preventDefault();
  const label = item.getAttribute('data-search');
  dropdownMenu.querySelectorAll('.dropdown-item-custom').forEach(el => el.classList.remove('active'));
  item.classList.add('active');
  openTab(label);
  setDropdownLabel(label);
});

// Listen for tab changes
tabBar.addEventListener('shown.bs.tab', (e) => {
  const label = e.target.textContent.trim().split('\n')[0].trim();
  toggleSidebarControls(label);
  setDropdownLabel(label);
  dropdownMenu.classList.add('selected-tab');
});

// Remove highlight when no tab is selected (optional, e.g. on tab close)
tabBar.addEventListener('click', (e) => {
  if (e.target.matches('[data-close]')) {
    setTimeout(() => {
      if (!tabBar.querySelector('.nav-link.active')) {
        dropdownMenu.classList.remove('selected-tab');
      }
    }, 100);
  }
});

// SigLIP2 Search event handler
document.addEventListener('click', function(e) {
  if (e.target && e.target.id === 'siglip2-search-btn') {
    const tabPane = e.target.closest('.tab-pane');
    const input = tabPane.querySelector('input[type="text"]');
    const query = input.value.trim();
    const kValueInput = document.getElementById('kValue');
    let top_k = 30;
    if (kValueInput && kValueInput.value) top_k = parseInt(kValueInput.value) || 30;
    if (!query) return;

    e.target.disabled = true;
    e.target.textContent = 'Searching...';
    fetch('/siglip2_search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: `query=${encodeURIComponent(query)}&top_k=${top_k}`
    })
    .then(res => res.text())
    .then(html => {
      const resultsDiv = tabPane.querySelector('.search-results');
      resultsDiv.innerHTML = html;
    })
    .catch(() => {
      tabPane.querySelector('.search-results').innerHTML = '<div class="text-danger">Error searching.</div>';
    })
    .finally(() => {
      e.target.disabled = false;
      e.target.textContent = 'Search';
    });
  }
});

// === Card interactions =======================================================

// Toggle selection by clicking the whole card (ignore double-clicks)
document.addEventListener('click', (e) => {
  const card = e.target.closest('.img-card.clickable');
  if (!card) return;

  // Ignore clicks on the Q icon
  if (e.target.closest('.img-q-icon')) return;

  card.classList.toggle('is-selected');
});

// Open the frame window on double-click
document.addEventListener('dblclick', (e) => {
  const imgArea = e.target.closest('.img-card.clickable .img-wrapper, .img-card.clickable img');
  
  if (e.target.closest('.img-q-icon')) return;

  
  if (!imgArea) return;                         // dblclick must be on the image
  const card = imgArea.closest('.img-card.clickable');
  if (!card) return;

  const videoId = card.getAttribute('data-video-id');
  const frame   = card.getAttribute('data-frame');
  if (!videoId || !frame) return;

  fetch(`/frame_window?video_id=${encodeURIComponent(videoId)}&frame=${encodeURIComponent(frame)}&w=50`)
    .then(res => res.text())
    .then(html => {
      const container = document.getElementById('frameModalContent');
      container.innerHTML = html;
      new bootstrap.Modal(document.getElementById('frameModal')).show();
    })
    .catch(() => alert('Failed to load frames.'));
});

// === Modal interactions ======================================================

// In the modal: clicking a thumbnail swaps the big image and styling.
// - Origin (double-clicked) frame stays ORANGE (thumb has data-origin="true")
// - Currently chosen frame becomes GREEN, and title/badge update
document.addEventListener('click', (e) => {
  const thumb = e.target.closest('#frameModal .frame-thumb');
  if (!thumb) return;

  const frameNum = thumb.dataset.frame;
  const viewer   = document.getElementById('frame-viewer-img');
  const badge    = document.getElementById('viewerBadge');
  const titleNum = document.getElementById('currentFrameNumber');
  const vbox     = document.getElementById('viewerBox');

  if (viewer) viewer.src = thumb.src;

  // highlight the active thumbnail (green), keep origin thumb orange
  document.querySelectorAll('#frameModal .frame-thumb').forEach(t => t.classList.remove('active'));
  thumb.classList.add('active');

  // viewer highlight: origin (orange) vs selected (green)
  if (thumb.dataset.origin === 'true') {
    vbox.classList.add('is-origin');
    vbox.classList.remove('is-selected');
  } else {
    vbox.classList.remove('is-origin');
    vbox.classList.add('is-selected');
  }

  // update texts
  if (badge)    badge.textContent = `Frame ${frameNum}`;
  if (titleNum) titleNum.textContent = frameNum;
});

// Initialize with SigLIP2 Search tab open
openTab('SigLIP2 Search');
