#!/usr/bin/env bash
set -e

echo "🚀 IHCMX Optimizer Pro – Installer"

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Linux*)  PLATFORM=Linux;;
  Darwin*) PLATFORM=Mac;;
  CYGWIN*|MINGW*) PLATFORM=Windows;;
  *)       echo "Unsupported OS: $OS"; exit 1;;
esac

echo "Detected platform: $PLATFORM"

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ Installation complete! Activate the environment with:"
echo "source venv/bin/activate   # Linux/Mac"
echo "venv\\Scripts\\activate      # Windows"
