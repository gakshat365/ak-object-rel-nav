#!/bin/bash
if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
