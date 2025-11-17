@echo off
ECHO ==================================
ECHO  Starting Door Listener...
ECHO ==================================

ECHO Activating Python environment...

:: This runs the batch file to activate your venv
CALL .\my_project_env\Scripts\activate.bat

ECHO Environment activated. Running the script...
ECHO (This window will stay open. Close it to stop the script.)

:: This runs your Python script
python run_detector.py