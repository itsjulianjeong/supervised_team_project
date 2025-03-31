@echo off
git config filter.nbstrip_full.clean "jq '.cells[].outputs = [] | .cells[].execution_count = null | .metadata = {} | .cells[].metadata = {}'"
echo Jupyter Notebook Git 필터가 설정되었습니다.
pause