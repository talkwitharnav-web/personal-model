@echo off
REM Resume the v2 fine-tune after a reboot.
REM
REM Why this file exists: on 2026-08-19 another workload took the machine's RAM to 91 MB, the
REM fine-tune died inside cuBLAS at step 9,100, and the NVIDIA driver was left wedged - nvidia-smi
REM stopped answering and the training process could not be killed. That needs a reboot, and no
REM amount of retrying fixes it. Nothing was lost: step 9,000 is on disk at validation 1.5925 and
REM the resume path is verified (tests\verify_finetune_resume.py, 11/11).
REM
REM Just double-click this after logging back in. The watchdog restarts training from step 9,000,
REM waits for free RAM and a responsive GPU before each attempt, and exits when the run completes.

cd /d "%~dp0"
echo Starting the v2 fine-tune watchdog (resumes from step 9,000)...
echo Progress:  logs\finetune_v2.log
echo Watchdog:  tests\finetune_watchdog.log
echo.
".venv\Scripts\python.exe" -u -m tools.finetune_watchdog
pause
