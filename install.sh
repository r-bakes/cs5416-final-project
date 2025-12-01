#!/bin/bash

echo "Running install.sh..."

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

echo "Installation complete!"

