@echo off
REM ============================================================
REM setup_env.bat — One-click project environment setup
REM Uses Python 3.12 from the official python.org installer
REM Run from: c:\Users\gudae\Desktop\Learn_ml\
REM ============================================================

SET PYTHON312=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
SET VENV_DIR=venv312

echo ============================================================
echo   Exoplanet Detection Pipeline — Environment Setup
echo ============================================================

echo.
echo [1/5] Checking Python 3.12...
"%PYTHON312%" --version
IF ERRORLEVEL 1 (
    echo ERROR: Python 3.12 not found at %PYTHON312%
    echo Please install from https://python.org/downloads
    pause
    exit /b 1
)

echo.
echo [2/5] Creating virtual environment...
IF EXIST "%VENV_DIR%" (
    echo     Virtual environment already exists, skipping creation.
) ELSE (
    "%PYTHON312%" -m venv %VENV_DIR%
    echo     Created: %VENV_DIR%\
)

echo.
echo [3/5] Upgrading pip...
%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip --quiet

echo.
echo [4/5] Installing all packages (may take 5-10 min on first run)...
%VENV_DIR%\Scripts\pip.exe install -r requirements.txt
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Package installation failed. Check error messages above.
    pause
    exit /b 1
)

echo.
echo [5/5] Registering Jupyter kernel as "Python (Exoplanet ML)"...
%VENV_DIR%\Scripts\python.exe -m ipykernel install --user --name exoplanet --display-name "Python (Exoplanet ML)"

echo.
echo ============================================================
echo   Setup complete!
echo ============================================================
echo.
echo To start Jupyter Notebook:
echo     %VENV_DIR%\Scripts\jupyter notebook
echo.
echo Or run the pipeline on a star:
echo     %VENV_DIR%\Scripts\python pipeline.py --tic_id 261136679
echo.
pause
