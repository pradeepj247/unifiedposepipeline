"""
Download and setup required models for both ViTPose and RTMLib pipelines
"""

import os
import sys
from pathlib import Path
import urllib.request
from tqdm import tqdm

# Model URLs
MODELS = {
    'vitpose': {
        'vitpose-b': {
            'url': 'https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK?e=Q1uZKs',
            'filename': 'vitpose-b-coco.pth',
            'size': '~90MB'
        },
        'vitpose-l': {
            'url': 'https://1drv.ms/u/s!AimBgYV7JjTlgSd9k_kuktPtiP4F?e=K7DGYT',
            'filename': 'vitpose-l-coco.pth',
            'size': '~320MB'
        },
    },
    'yolo': {
        'yolov8n': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'filename': 'yolov8n.pt',
            'size': '~6MB'
        },
        'yolov8m': {
            'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'filename': 'yolov8m.pt',
            'size': '~50MB'
        },
    },
    'rtmlib': {
        # RTMLib models are downloaded automatically by the library
        'info': 'RTMLib models will be downloaded automatically on first use'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download a file with progress bar"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"✓ {output_path.name} already exists, skipping...")
        return True
    
    print(f"Downloading {output_path.name}...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        print(f"✓ Downloaded {output_path.name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {output_path.name}: {e}")
        return False


def main():
    """Main download function"""
    print("=" * 60)
    print("Unified Pose Estimation - Model Downloader")
    print("=" * 60)
    
    # Get models directory
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    print(f"\nModels will be saved to: {models_dir.absolute()}\n")
    
    # Download YOLO models (essential for both pipelines)
    print("\n📦 Downloading YOLO Detection Models...")
    print("-" * 60)
    yolo_dir = models_dir / 'yolo'
    yolo_dir.mkdir(exist_ok=True)
    
    for model_name, model_info in MODELS['yolo'].items():
        output_path = yolo_dir / model_info['filename']
        print(f"\n{model_name} ({model_info['size']})")
        download_file(model_info['url'], output_path)
    
    # ViTPose models
    print("\n\n📦 ViTPose Models")
    print("-" * 60)
    print("Note: ViTPose models are large. Download them as needed.")
    print("For now, please visit the ViTPose repository to download models:")
    print("https://github.com/ViTAE-Transformer/ViTPose")
    
    vitpose_dir = models_dir / 'vitpose'
    vitpose_dir.mkdir(exist_ok=True)
    print(f"Place ViTPose models in: {vitpose_dir.absolute()}")
    
    # RTMLib info
    print("\n\n📦 RTMLib Models")
    print("-" * 60)
    print(MODELS['rtmlib']['info'])
    
    rtmlib_dir = models_dir / 'rtmlib'
    rtmlib_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Download ViTPose models manually from the repository")
    print("2. Run the demo scripts in the 'demos/' directory")
    print("3. Check the notebooks for detailed examples")


if __name__ == '__main__':
    main()
