@echo off
SETLOCAL

REM Set project directory (backend folder)
set PROJECT_DIR=%~dp0backend

echo ==============================
echo Installing dependencies
echo ==============================

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
if exist "%PROJECT_DIR%\requirements.txt" (
    pip install -r "%PROJECT_DIR%\requirements.txt"
) else (
    echo requirements.txt not found at %PROJECT_DIR%
    pause
    exit /b
)

echo ==============================
echo Running app.py
echo ==============================

if exist "%PROJECT_DIR%\app.py" (
    python "%PROJECT_DIR%\app.py"
) else (
    echo app.py not found at %PROJECT_DIR%
)

echo ==============================
echo Done
echo ==============================

pause