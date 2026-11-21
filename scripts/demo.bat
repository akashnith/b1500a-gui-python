@echo off
cd /d "%~dp0.."
echo Simulated WGFMU LTP/LTD measurement (no instrument needed)...
echo.
python -m b1500_studio run synapse --demo --out results --png
echo.
pause
