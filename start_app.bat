@echo off
rem ============================================================
rem  Fake News Detector - one-click launcher
rem  Double-click this file to start the app and open it in
rem  your browser. Close the window that appears to stop the app.
rem ============================================================

title Fake News Detector
cd /d "%~dp0"

rem --- already running? Just open the browser instead of failing ---
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:5000/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo App is already running - opening it in your browser.
    start "" http://127.0.0.1:5000
    timeout /t 2 >nul
    exit /b 0
)

rem --- first run ever? Train the model (skips silently if model exists) ---
if not exist "model\model.pkl" (
    echo First run: training the model. This takes a minute or two...
    py train_model.py
    if errorlevel 1 (
        echo.
        echo Training failed. Check the message above.
        pause
        exit /b 1
    )
)

echo.
echo  ============================================
echo   Fake News Detector is starting...
echo   Your browser will open at 127.0.0.1:5000
echo.
echo   Keep this window OPEN while using the app.
echo   Close this window (or press Ctrl+C) to stop.
echo  ============================================
echo.

rem --- launch server in the background, then open the browser ---
start "fakenews-server" /min py app.py
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:5000

rem --- keep the window alive showing a friendly message ---
echo You can close this window - the app keeps running in the
echo "fakenews-server" taskbar window. Re-run this file any time.
timeout /t 5 >nul
exit /b 0
