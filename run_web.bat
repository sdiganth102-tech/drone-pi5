@echo off
cd /d "%~dp0"
echo ========================================================
echo   Launching Drone Flood Rescue Web Mission Control...
echo   Open in browser: http://localhost:5000
echo ========================================================
python web_app.py
pause
