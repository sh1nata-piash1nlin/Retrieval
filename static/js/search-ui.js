// Icons for each search type
const ICONS = {
  "Single Search": "fa-solid fa-magnifying-glass",
  "Fusion Search": "fa-regular fa-circle",
  "Local Search": "fa-regular fa-gem",
  "Group Search": "fa-solid fa-users",
  "Hierarchy Search": "fa-solid fa-layer-group",
  "Subtitle Match": "fa-solid fa-align-left",
  "OCR Match": "fa-regular fa-star",
  "Similar Image Search": "fa-regular fa-image",
  "Similar Frame Search": "fa-solid fa-image"
};

// HTML templates for each tab (replace with your real markup / forms)
const TEMPLATES = {
  "Single Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-success" id="single-search-btn">Search</button>
      </div>
      <div></div>
    </div>
    <div class="search-results">
      <!-- Search results will be loaded here -->
    </div>
  `,
  "Hierarchy Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm..." />
        <button class="btn btn-primary" id="hierarchy-search-btn">Search</button>
      </div>
      <div></div>
    </div>
    <div class="hierarchy-search-results">
      <!-- Hierarchy search results will be loaded here -->
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
  if (label === 'Hierarchy Search') {
    kRow.style.display = 'none';
    hierarchyKRow.style.display = '';
  } else {
    kRow.style.display = '';
    hierarchyKRow.style.display = 'none';
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

// Event listener for the single search button (using event delegation)
document.addEventListener('click', async (e) => {
  if (e.target && e.target.id === 'single-search-btn') {
    const searchInput = e.target.previousElementSibling;
    const query = searchInput.value;
    const resultsContainer = e.target.closest('.tab-pane').querySelector('.search-results');

    const formData = new FormData();
    formData.append('query', query);

    try {
      const response = await fetch('/search', {
        method: 'POST',
        body: formData
      });
      resultsContainer.innerHTML = await response.text();
    } catch (error) {
      console.error('Error fetching search results:', error);
      resultsContainer.innerHTML = '<p class="text-danger">Error loading results.</p>';
    }
  }
  // Hierarchy Search event
  if (e.target && e.target.id === 'hierarchy-search-btn') {
    const searchInput = e.target.closest('.search-bar').querySelector('input[type="text"]');
    const query = searchInput.value;
    const k = document.querySelector('#hierarchyKRowContainer #kValue').value;
    const k1 = document.querySelector('#hierarchyKRowContainer #hierarchyK1Sidebar').value;
    const resultsContainer = e.target.closest('.tab-pane').querySelector('.hierarchy-search-results');

    const formData = new FormData();
    formData.append('query', query);
    formData.append('k', k);
    formData.append('k1', k1);

    try {
      const response = await fetch('/hierarchy_search', {
        method: 'POST',
        body: formData
      });
      resultsContainer.innerHTML = await response.text();
    } catch (error) {
      console.error('Error fetching hierarchy search results:', error);
      resultsContainer.innerHTML = '<p class="text-danger">Error loading results.</p>';
    }
  }
});

// Initialize with Single Search tab open
openTab('Single Search');
