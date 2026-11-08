#!/bin/bash
set -e
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt || pip install -q --user -r requirements.txt
cd src && python experiments.py
