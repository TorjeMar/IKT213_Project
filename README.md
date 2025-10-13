# IKT213 Project

Vehicle detection, license plate recognition, and speed measurement using machine vision vision.

## What's Inside

- `main.ipynb` - License plate detection from images
- `platefinder.ipynb` - Vehicle detection and tracking
- `speedOmeter.ipynb` - Speed measurement from videos

## Quick Start

### 1. Install Tesseract OCR

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**  
Download from https://github.com/UB-Mannheim/tesseract/wiki

### 2. Setup Python Environment

```bash
# Clone the repo
git clone https://github.com/TorjeMar/IKT213_Project.git
cd IKT213_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Tesseract Path

In `main.ipynb`, update the line with your Tesseract path.


### 4. Run the Notebooks

Open in VS Code or start Jupyter:
```bash
jupyter notebook
```
