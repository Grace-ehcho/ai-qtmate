#!/bin/bash
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
exec python3 -m streamlit run "$(dirname "$0")/app.py" "$@"
