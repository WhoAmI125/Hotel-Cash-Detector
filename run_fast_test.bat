@echo off
REM Fast test - Only processes specific moments (2-5 minutes instead of 30+)
echo ========================================
echo FAST TEST - Specific Moments Only
echo ========================================
echo.
echo This test extracts only the time ranges where events occur
echo and processes them instead of the full 60-minute video.
echo.
echo Expected time: 2-5 minutes
echo.

REM Clear Python cache first
echo Clearing Python cache...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo Cache cleared!
echo.

REM Run the fast test
echo Running fast test on specific moments...
python test_specific_moments.py

echo.
echo ========================================
echo Test complete! Check results above.
echo ========================================
echo.
echo If you see MULTIPLE violence clips (5+), the fix is working!
echo If you see only 1 violence clip, the fix needs more work.
echo.
pause
