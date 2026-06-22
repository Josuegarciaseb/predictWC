@echo off
REM Levanta el dashboard del Predictor Mundial 2026
cd /d "%~dp0.."
echo Abriendo dashboard en http://localhost:8000/web/
start "" http://localhost:8000/web/
py -m http.server 8000
