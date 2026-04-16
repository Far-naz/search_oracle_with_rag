@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
	py -3 -m streamlit run "%~dp0app.py"
) else (
	python -m streamlit run "%~dp0app.py"
)
