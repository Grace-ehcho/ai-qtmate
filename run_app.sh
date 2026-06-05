#!/bin/bash
# UTF-8 환경 변수를 Python 시작 전에 설정 — 이 방법이 가장 확실함
# (Python 프로세스 내부에서 os.environ을 설정하면 현재 프로세스에는 적용 안 됨)
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

exec python3 -m streamlit run "$(dirname "$0")/app.py" "$@"
