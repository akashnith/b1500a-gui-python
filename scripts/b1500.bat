@echo off
rem Forward any CLI command, e.g.:  wgfmu run synapse --demo
cd /d "%~dp0.."
python -m b1500_studio %*
