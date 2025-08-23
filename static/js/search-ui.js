// Icons for each search type
const ICONS = {
  "InternVideo2 Search": "fa-solid fa-video",
  "blip2 Search": "fa-solid fa-robot",
  "bge-m3 Search": "fa-solid fa-brain",
  "FDP Search": "fa-solid fa-database",
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
      </div>
      <div></div>
    </div>
    <div class="search-results">
      <!-- Search results will be loaded here -->
    </div>
  `,
  "blip2 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-primary" id="blip2-search-btn">Search</button>
      </div>
      <div></div>
    </div>
    <div class="search-results">
      <!-- Search results will be loaded here -->
    </div>
  `,
  "bge-m3 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-primary" id="bge-m3-search-btn">Search</button>
      </div>
      <div></div>
    </div>
    <div class="search-results">
      <!-- Search results will be loaded here -->
    </div>
  `,
  "FDP Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm FDP..." />
        <button class="btn btn-info" id="fdp-search-btn">Search</button>
      </div>
      <div></div>
    </div>
    <div class="search-results">
      <!-- FDP search results will be loaded here -->
    </div>
  `,
  "OCR Match": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="text" class="form-control" placeholder="Nhập nội dung OCR cần tìm…" />
      <button class="btn btn-primary">Search OCR</button>
    </div>
  `,
  "Subtitle Match": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="text" class="form-control" placeholder="Nhập câu phụ đề cần tìm…" />
      <button class="btn btn-primary">Search Subtitle</button>
    </div>
  `,
  "Similar Image Search": () => `
    <input type="file" class="form-control mb-3" accept="image/*" />
    <button class="btn btn-primary">Find Similar Images</button>
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
    // Show/hide sidebar controls
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

  // Show/hide sidebar controls
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
      kRow.style.display = 'block';
      break;
    case 'Hierarchy Search':
      hierarchyKRow.style.display = 'block';
      break;
    case 'Similar Image Search':
      kRow.style.display = 'block';
      groupImagesContainer.style.display = 'block';
      break;
    case 'Subtitle Match':
      kRow.style.display = 'block';
      break;
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
  // Mark active item
  dropdownMenu.querySelectorAll('.dropdown-item-custom').forEach(el => el.classList.remove('active'));
  item.classList.add('active');
  // Open / activate tab and update button text
  openTab(label);
  setDropdownLabel(label);
});

// Listen for tab changes
tabBar.addEventListener('shown.bs.tab', (e) => {
  const label = e.target.textContent.trim().split('\n')[0].trim();
  toggleSidebarControls(label);
  setDropdownLabel(label); // Update dropdown label to match active tab
  // Highlight dropdown when tab is selected
  dropdownMenu.classList.add('selected-tab');
});

// Remove highlight when no tab is selected (optional, e.g. on tab close)
tabBar.addEventListener('click', (e) => {
  if (e.target.matches('[data-close]')) {
    setTimeout(() => {
      // If no tab is active, remove highlight
      if (!tabBar.querySelector('.nav-link.active')) {
        dropdownMenu.classList.remove('selected-tab');
      }
    }, 100);
  }
});

// Close tab via delegation
tabBar.addEventListener('click', (e) => {
  if (e.target.matches('[data-close]')) {
    e.stopPropagation();
    const id = e.target.getAttribute('data-close');
    const link = document.getElementById(`tab-${id}-link`);
    const pane = document.getElementById(`tab-${id}`);
    const wasActive = link.classList.contains('active');

    // Remove elements
    link.parentElement.remove();
    pane.remove();

    if (wasActive) {
      // Activate neighbor tab if any
      const next = tabBar.querySelector('.nav-link');
      if (next) new bootstrap.Tab(next).show();
    }
  }
});

// ...existing code...

// You can add event listeners for new search buttons here if needed


// Initialize with InternVideo2 Search tab open
openTab('InternVideo2 Search');
