#!/bin/bash
set -e

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install requirements with pre-built wheels
pip install -r requirements.txt --no-cache-dir --prefer-binary

# Run the model caching script
python cache_model.py