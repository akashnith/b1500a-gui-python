@echo off
cd /d "%~dp0.."
echo Simulated bipolar resistive-switching I-V (no instrument needed)...
echo.
python -m b1500_studio smu protocol iv --set sweep_v_pos_v=1.2 --set sweep_v_neg_v=1.5 --set sweep_step_v=0.02 --set step_dwell_s=0 --common-ch 2 --out results --png
echo.
pause
