#!/bin/bash

# Get the absolute path to the server directory
SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Add server directory to PYTHONPATH
export PYTHONPATH="$SERVER_DIR:$PYTHONPATH"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the database management command directly
python "$SCRIPT_DIR/manage_db.py" "$@" 