@echo off
cd /d "%~dp0"
call venv\Scripts\activate
python dataset_studio_gui.py
pause

