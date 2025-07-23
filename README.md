# Video Frame Retrieval System

This project is a web-based video frame retrieval system that provides various search functionalities including single search, hierarchy search, subtitle match, and similar image search.

## Prerequisites

- Python 3.8 or higher
- Conda (for environment management)

## Installation

1. Clone the repository
```bash
git clone https://github.com/sh1nata-piash1nlin/Retrieval.git
cd Retrieval
```

2. Create and activate a Conda environment
```bash
conda create -n preAIC python=3.9
conda activate retrieval
```

3. Install the required packages
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── app.py              
├── requirements.txt    
├── static/
│   ├── css/           
│   ├── js/            
│   └── keyframe/      
└── templates/         
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

3. Open your web browser and navigate to `http://localhost:2714`

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


