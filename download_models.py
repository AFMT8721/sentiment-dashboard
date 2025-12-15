"""Download model files from GitHub if they don't exist locally."""
import os
import requests
from pathlib import Path

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/AFMT8721/sentiment-dashboard/main/models/"
MODEL_FILES = [
    "lstm_model.h5",
    "tokenizer.pickle",
    "best_logistic_regression_model.pickle"
]

def download_file(url, destination):
    """Download a file from URL to destination."""
    print(f"Downloading {url} to {destination}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ Downloaded {destination}")

def ensure_models():
    """Ensure all required model files exist."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    for model_file in MODEL_FILES:
        model_path = models_dir / model_file
        if not model_path.exists():
            url = GITHUB_RAW_BASE + model_file
            try:
                download_file(url, model_path)
            except Exception as e:
                print(f"✗ Failed to download {model_file}: {e}")
                raise

if __name__ == "__main__":
    ensure_models()
