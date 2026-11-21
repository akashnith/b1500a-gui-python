@echo off
cd /d "%~dp0.."
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest %*
echo.
pause
