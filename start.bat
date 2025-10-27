@echo off
REM Simple script to set up the environment and run the LLM Studio app on Windows.

REM --- Configuration ---
SET PYTHON_CMD=python
SET VENV_DIR=venv

REM --- Main Script ---
echo INFO: Launching HF LLM Studio Setup...

REM 1. Check if Python is installed
%PYTHON_CMD% --version >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is not found in PATH. Please install Python 3 and ensure it's added to your PATH.
    pause
    exit /b 1
)

REM 2. Check if venv directory exists
if not exist "%VENV_DIR%\" (
    echo INFO: Python virtual environment not found. Creating one...
    %PYTHON_CMD% -m venv %VENV_DIR%
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create the virtual environment.
        pause
        exit /b 1
    )
    echo INFO: Virtual environment created in '%VENV_DIR%\\'.
)

REM 3. Activate the virtual environment
echo INFO: Activating the virtual environment...
call "%VENV_DIR%\\Scripts\\activate.bat"

REM 4. Install dependencies
echo INFO: Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies. Please check requirements.txt and your internet connection.
    call "%VENV_DIR%\\Scripts\\deactivate.bat"
    pause
    exit /b 1
)

REM 5. Run the Streamlit application
echo INFO: All set! Starting the HF LLM Studio...
echo INFO: You can close this window to stop the application.
streamlit run app.py

REM 6. Deactivate the virtual environment upon closing the app
echo INFO: Application closed.
call "%VENV_DIR%\\Scripts\\deactivate.bat"
pause
