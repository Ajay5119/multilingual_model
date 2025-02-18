#!/bin/bash

# Update package lists
apt-get update

# Install system dependencies
apt-get install -y ffmpeg

# Install Python dependencies
pip install --no-cache-dir -r requirements.txt

# Explicitly install langchain_ollama
pip install --no-cache-dir langchain_ollama
