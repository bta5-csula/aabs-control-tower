@echo off
REM ============================================================
REM AABS Control Tower v8.3 - Windows Startup Script
REM ============================================================

echo ============================================================
echo AABS Control Tower v8.3
echo Enterprise Decision Intelligence Platform
echo ============================================================
echo.

REM 1. Find the best Python command
echo [1/4] Finding Python (Recommended: 3.10-3.12)...
set PYTHON_CMD=
py -3.12 --version >nul 2>&1 && set PYTHON_CMD=py -3.12
if not defined PYTHON_CMD (
    py -3.11 --version >nul 2>&1 && set PYTHON_CMD=py -3.11
)
if not defined PYTHON_CMD (
    py -3.10 --version >nul 2>&1 && set PYTHON_CMD=py -3.10
)
if not defined PYTHON_CMD (
    python --version >nul 2>&1 && set PYTHON_CMD=python
)

if not defined PYTHON_CMD (
    echo ERROR: Python not found. Please install Python 3.10-3.12.
    pause
    exit /b 1
)

%PYTHON_CMD% --version
echo Using command: %PYTHON_CMD%
echo.

REM 2. Set up Virtual Environment
echo [2/4] Setting up virtual environment...
if not exist ".venv" (
    echo Creating new virtual environment...
    %PYTHON_CMD% -m venv .venv
)
echo Virtual environment ready
echo.

REM 3. Install Dependencies
echo [3/4] Installing dependencies...
call .venv\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
echo Dependencies installed
echo.

REM 4. Launch Application
echo [4/4] Launching AABS Control Tower...
echo.
echo ============================================================
echo Starting server at http://localhost:8501
echo Press Ctrl+C to stop
echo ============================================================
echo.

streamlit run app.py --server.headless true --browser.gatherUsageStats false
