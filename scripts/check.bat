@echo off
cd /d "%~dp0.."
echo Diagnosing the B1500A / WGFMU connection...
echo.
python -m b1500_studio check %*
echo.
pause
