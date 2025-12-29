#!/usr/bin/env python3
"""
Setup script for Chocolate Counter
Creates necessary folders and verifies installation
"""

import os
from pathlib import Path
import subprocess
import sys


def create_directory_structure():
    """Create required folders for the counting system."""
    
    folders = [
        'videos/input',          # Place test videos here
        'output/videos',         # Processed videos with annotations
        'output/logs',           # CSV and JSON logs
        'output/debug_frames',   # Debug frames (optional)
    ]
    
    print("Creating directory structure...")
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {folder}")
    
    print("\nDirectory structure created successfully!")


def check_dependencies():
    """Check if required packages are installed."""
    
    required_packages = {
        'cv2': 'opencv-python',
        'yaml': 'pyyaml',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'ultralytics': 'ultralytics'
    }
    
    print("\nChecking dependencies...")
    missing = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print(f"  pip install {' '.join(missing)}")
        return False
    else:
        print("\nAll dependencies satisfied!")
        return True


def verify_model():
    """Check if trained model exists."""
    
    print("\nVerifying model file...")
    
    # Common model locations
    model_paths = [
        'runs/detect/train/weights/best.pt',
        'runs/detect/train2/weights/best.pt',
        'runs/detect/train3/weights/best.pt',
        'models/best.pt'
    ]
    
    found = None
    for path in model_paths:
        if Path(path).exists():
            found = path
            break
    
    if found:
        print(f"  ✓ Model found: {found}")
        print(f"\nUpdate config.yaml to use this path:")
        print(f"  paths:")
        print(f"    model: \"{found}\"")
        return True
    else:
        print("  ✗ Model not found in common locations")
        print("\nPlease update config.yaml with your model path:")
        print("  paths:")
        print("    model: \"<your-model-path>/best.pt\"")
        return False


def create_example_config():
    """Check if config.yaml exists."""
    
    if Path('config.yaml').exists():
        print("\n✓ config.yaml exists")
        return True
    else:
        print("\n✗ config.yaml not found")
        print("Please ensure config.yaml is in the project root directory")
        return False


def main():
    """Run setup checks."""
    
    print("="*70)
    print("Chocolate Counter - Setup & Verification")
    print("="*70)
    
    # Create folders
    create_directory_structure()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check model
    model_ok = verify_model()
    
    # Check config
    config_ok = create_example_config()
    
    # Final summary
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    
    if deps_ok and model_ok and config_ok:
        print("✓ All checks passed! You're ready to run the counter.")
        print("\nNext steps:")
        print("  1. Place a test video in: videos/input/")
        print("  2. Update config.yaml with the video filename")
        print("  3. Run: python counter.py")
    else:
        print("⚠ Some issues need attention. See messages above.")
    
    print("="*70)


if __name__ == "__main__":
    main()
