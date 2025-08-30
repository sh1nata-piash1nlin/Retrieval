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

// HTML templates for each tab
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
    <div class="search-results row row-cols-5 g-3">
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
    <div class="search-results row row-cols-5 g-3">
      <!-- Search results will be loaded here -->
    </div>
  `,
  "SigLIP2 Search": () => `
    <div class="d-flex justify-content-between align-items-center mb-4">
      <div></div>
      <div class="search-bar w-75 d-flex align-items-center gap-2">
        <input type="text" class="form-control" placeholder="Nhập mô tả tìm kiếm SigLIP2..." />
        <button class="btn btn-primary" id="siglip2-search-btn">Search</button>
      </div>
      <div></div>
    </div>
    <div class="search-results row row-cols-5 g-3">
      <!-- SigLIP2 search results will be loaded here -->
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
    <div class="search-results row row-cols-5 g-3">
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
    <div class="search-results row row-cols-5 g-3">
      <!-- FDP search results will be loaded here -->
    </div>
  `,
  "OCR Match": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="text" class="form-control" placeholder="Nhập nội dung OCR cần tìm…" />
      <button class="btn btn-primary" id="ocr-search-btn">Search OCR</button>
    </div>
    <div class="search-results row row-cols-5 g-3">
      <!-- OCR search results will be loaded here -->
    </div>
  `,
  "Subtitle Match": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="text" class="form-control" placeholder="Nhập câu phụ đề cần tìm…" />
      <button class="btn btn-primary" id="subtitle-search-btn">Search Subtitle</button>
    </div>
    <div class="search-results row row-cols-5 g-3">
      <!-- Subtitle search results will be loaded here -->
    </div>
  `,
  "Similar Image Search": () => `
    <div class="mb-3 d-flex gap-2">
      <input type="file" class="form-control" accept="image/*" id="image-upload" />
      <button class="btn btn-primary" id="image-search-btn">Find Similar Images</button>
    </div>
    <div class="search-results row row-cols-5 g-3">
      <!-- Image search results will be loaded here -->
    </div>
  `
};
function fetchNeighboringFrames(videoId, frameNum, modelType) {
  const formData = new FormData();
  formData.append('video_id', videoId);
  formData.append('frame_num', frameNum);
  formData.append('model_type', modelType);

  return fetch('http://172.17.0.3:2714/neighboring_frames', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        throw new Error(data.error);
      }
      return data.frames; 
    });
}

let frameModal = null;
let carouselListener = null;

// Function to render slider modal
function renderSlider(frames, initialIndex) {
  const carouselInner = document.getElementById('carouselInner');
  const frameInfo = document.getElementById('frameInfo');
  const carouselEl = document.getElementById('frameCarousel');
  const carousel = new bootstrap.Carousel(carouselEl, {
    interval: false,
    ride: false,
    wrap: false
  });

  carouselInner.innerHTML = '';

  // Render frames
  frames.forEach((frame, index) => {
    const isActive = index === initialIndex ? 'active' : '';
    const item = document.createElement('div');
    item.className = `carousel-item ${isActive}`;
    item.innerHTML = `
      <img src="${frame.image_url}" class="d-block w-100" alt="Frame ${frame.frame_num}">
    `;
    carouselInner.appendChild(item);
  });

  let currentFrame = frames[initialIndex];
  frameInfo.textContent = `Video: ${currentFrame.video_id}, Frame: ${currentFrame.frame_num}`;

  // Remove old listener to prevent duplicates
  if (carouselListener) {
    carousel.removeEventListener('slid.bs.carousel', carouselListener);
  }

  // Define and attach new listener
  carouselListener = (e) => {
    const activeIndex = e.to;
    currentFrame = frames[activeIndex];
    frameInfo.textContent = `Video: ${currentFrame.video_id}, Frame: ${currentFrame.frame_num}`;

    // Fetch more frames if at edges
    if (activeIndex === 0 || activeIndex === frames.length - 1) {
      const activeTab = document.querySelector('#resultTabs .nav-link.active');
      const modelType = activeTab ? activeTab.getAttribute('data-model') : "siglip2";

      fetchNeighboringFrames(currentFrame.video_id, currentFrame.frame_num, modelType)
        .then(newFrames => {
          const newIndex = newFrames.findIndex(f => f.frame_num === currentFrame.frame_num);
          renderSlider(newFrames, newIndex >= 0 ? newIndex : 0);
        })
        .catch(err => {
          console.error("Error fetching neighboring frames:", err);
        });
    }
  };
  carousel.addEventListener('slid.bs.carousel', carouselListener);

  // Show modal (only create once)
  if (!frameModal) {
    frameModal = new bootstrap.Modal(document.getElementById('frameSliderModal'));
  }
  frameModal.show();
}




// Function to render search results
function renderResults(results, container) {
  container.innerHTML = ''; // Clear previous results
  results.forEach(result => {
    const col = document.createElement('div');
    col.className = 'col';
    col.innerHTML = `
      <div class="img-card" style="border-color: ${result.border_color};">
        <img src="${result.image_url}" alt="Frame" />
        <div class="img-label" style="--img-label-color: ${result.border_color};">${result.video_id}</div>
        <div class="img-number">${result.frame_num}</div>
        ${result.score !== undefined ? `<div class="img-score">Score: ${result.score.toFixed(3)}</div>` : ''}
      </div>
    `;
    const imgCard = col.querySelector('.img-card');
    const imgElement = imgCard.querySelector('img'); // Define imgElement
    const labelElement = imgCard.querySelector('.img-label'); // Define labelElement
    const numberElement = imgCard.querySelector('.img-number'); // Define numberElement
    const activeTab = document.querySelector('#resultTabs .nav-link.active');
    const modelType = activeTab ? activeTab.getAttribute('data-model') : "siglip2";
    const topK = document.getElementById('kValue')?.value || 30;

    // Click on image for similarity search
    imgElement.addEventListener('click', (e) => {
      e.stopPropagation(); // Prevent event from bubbling to parent
      fetch(result.image_url)
        .then(res => res.blob())
        .then(blob => {
          const file = new File([blob], "clicked.jpg", { type: blob.type });
          handleImageSearch(file, modelType, topK, container);
        })
        .catch(err => {
          console.error("Error fetching clicked image:", err);
          alert("Could not load image for search.");
        });
    });

    // Click on label or frame number for neighboring frames
    [labelElement, numberElement].forEach(element => {
      element.addEventListener('click', (e) => {
        e.stopPropagation();
        fetchNeighboringFrames(result.video_id, result.frame_num, modelType)
          .then(frames => {
            let initialIndex = frames.findIndex(f => f.frame_num === result.frame_num);
            if (initialIndex === -1) {
              initialIndex = 0;
            }
            renderSlider(frames, initialIndex);
          })
          .catch(err => {
            console.error("Error fetching neighboring frames:", err);
            alert("Could not load neighboring frames.");
          });
      });
    });

    container.appendChild(col);
  });
}
// Function to handle text-based search
function handleTextSearch(query, modelType, topK, resultsContainer) {
  const formData = new FormData();
  formData.append('query', query);
  formData.append('top_k', topK);
  formData.append('model_type', modelType);

  fetch('http://172.17.0.3:2714/text_search', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }
      renderResults(data.results, resultsContainer);
    })
    .catch(error => {
      console.error('Search error:', error);
      alert('Search failed. Please try again.');
    });
}

// Function to handle image-based search
function handleImageSearch(file, modelType, topK, resultsContainer) {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('top_k', topK);
  formData.append('model_type', modelType);

  fetch('http://172.17.0.3:2714/image_search', {
    method: 'POST',
    body: formData
  })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }
      renderResults(data.results, resultsContainer);
    })
    .catch(error => {
      console.error('Search error:', error);
      alert('Search failed. Please try again.');
    });
}

// Function to load initial random results
function loadInitialResults(resultsContainer) {
  fetch('http://172.17.0.3:2714/search')
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert(`Error: ${data.error}`);
        return;
      }
      renderResults(data.results, resultsContainer);
    })
    .catch(error => {
      console.error('Initial load error:', error);
      alert('Failed to load initial results.');
    });
}

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

  // Show/hide sidebar controls
  toggleSidebarControls(label);

  // Load initial results for InternVideo2 Search
  if (label === 'InternVideo2 Search') {
    const resultsContainer = pane.querySelector('.search-results');
    loadInitialResults(resultsContainer);
  }

  // Add event listeners for search buttons
  const searchBtn = pane.querySelector('button[id$="-search-btn"]');
  const input = pane.querySelector('input[type="text"], input[type="file"]');
  const kValueInput = document.getElementById('kValue');
  const resultsContainer = pane.querySelector('.search-results');

  if (searchBtn && input && resultsContainer) {
    if (label === 'Similar Image Search') {
      searchBtn.addEventListener('click', () => {
        const file = input.files[0];
        if (!file) {
          alert('Please select an image.');
          return;
        }
        const topK = kValueInput ? kValueInput.value : 30;
        handleImageSearch(file, 'siglip', topK, resultsContainer); // Adjust model_type as needed
      });
    } else {
      searchBtn.addEventListener('click', () => {
        const query = input.value.trim();
        if (!query) {
          alert('Please enter a search query.');
          return;
        }
        const topK = kValueInput ? kValueInput.value : 30;
        const modelType = {
          'InternVideo2 Search': 'internvideo2',
          'blip2 Search': 'blip2', // Map to available model; adjust as needed
          'SigLIP2 Search': 'siglip2',
          'bge-m3 Search': 'siglip', // Map to available model; adjust as needed
          'FDP Search': 'fdp',
          'OCR Match': 'siglip', // Map to available model; adjust as needed
          'Subtitle Match': 'siglip' // Map to available model; adjust as needed
        }[label] || 'blip2';
        handleTextSearch(query, modelType, topK, resultsContainer);
      });
    }
  }
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

function setDropdownLabel(label) {
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
  setDropdownLabel(label);
  dropdownMenu.classList.add('selected-tab');
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
      else {
        dropdownMenu.classList.remove('selected-tab');
        setDropdownLabel('Single Search'); // Reset to default
      }
    }
  }
});

// Initialize with InternVideo2 Search tab open
openTab('InternVideo2 Search');