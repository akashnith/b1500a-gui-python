@echo off
cd /d "%~dp0.."
start "" pythonw -m b1500_studio gui %*
