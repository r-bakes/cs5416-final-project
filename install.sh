#!/bin/bash

echo "Running install.sh..."

python3 -m virtualenv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

echo "Installation complete!"

