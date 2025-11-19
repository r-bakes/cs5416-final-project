#!/bin/bash

echo "Running install.sh..."

python3 -m venv .
source .venv/bin/activate
python3 pip install -r requirements.txt

echo "Installation complete!"

