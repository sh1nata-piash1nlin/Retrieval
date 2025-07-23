# Video Frame Retrieval System

This project is a web-based video frame retrieval system that provides various search functionalities including single search, hierarchy search, subtitle match, and similar image search.

## Features

- Single Search
- Hierarchy Search
- Subtitle Match Search
- Similar Image Search
- Customizable K-value settings
- Group images per video configuration

## Prerequisites

- Python 3.8 or higher
- Conda (for environment management)

## Installation

1. Clone the repository
```bash
git clone https://github.com/sh1nata/Retrieval.git
cd Retrieval
```

2. Create and activate a Conda environment
```bash
conda create -n retrieval python=3.8
conda activate retrieval
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── app.py              # Flask application main file
├── requirements.txt    # Python dependencies
├── static/
│   ├── css/           # CSS styles
│   ├── js/            # JavaScript files
│   └── keyframe/      # Directory for keyframe images (not included in repo)
└── templates/         # HTML templates
```

## Usage

1. Activate the Conda environment (if not already activated)
```bash
conda activate retrieval
```

2. Run the Flask application
```bash
python app.py
```

3. Open your web browser and navigate to `http://localhost:5000`

## Configuration

- **K Value**: Adjust the number of results returned (default: 30)
- **K1 Value** (Hierarchy Search): Configure the first-level results (default: 5)
- **Images per Video**: Set the number of images displayed per video (default: 4)

## Note

The `static/keyframe` directory is not included in the repository due to size constraints. You'll need to add your own keyframe images in the following structure:

```
static/keyframe/
└── Keyframes_L01/
    └── keyframes/
        └── L01_V001/
            ├── 001.jpg
            ├── 002.jpg
            └── ...
```

## Contributing

Feel free to submit issues and enhancement requests.
