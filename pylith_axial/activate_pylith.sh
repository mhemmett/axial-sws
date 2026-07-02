#!/bin/bash
# Source this file to activate the PyLith 5.0.1 environment:
#   source activate_pylith.sh

export PYLITH_DIR=/Users/mhemmett/Downloads/pylith-5.0.1-macOS-12.0-arm64
export PATH=$PYLITH_DIR/bin:$PATH
export PYTHONHOME=$PYLITH_DIR
export DYLD_LIBRARY_PATH=$PYLITH_DIR/lib:$DYLD_LIBRARY_PATH
export PYTHONPATH=$PYLITH_DIR/lib/python3.12/site-packages

# Convenience alias
alias pylith="$PYLITH_DIR/bin/nemesis $PYLITH_DIR/bin/pylith"

echo "PyLith 5.0.1 environment active."
echo "Run simulations with: pylith pylithapp.cfg ..."
