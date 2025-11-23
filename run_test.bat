@echo off
REM Quick test batch file - runs detection on specific video
echo ========================================
echo Quick Test - Violence Split Fix (3s gap)
echo ========================================
echo.

REM Clear Python cache first
echo Clearing Python cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo Cache cleared!
echo.

REM Run the test (full video - will take ~15-30 minutes)
echo Running detection test on FULL VIDEO...
echo This will take 15-30 minutes. Please be patient.
echo.
echo Press Ctrl+C to cancel if needed.
echo.
python test_quick_run.py

echo.
echo ========================================
echo Test complete! Check results above.
echo ========================================
pause
