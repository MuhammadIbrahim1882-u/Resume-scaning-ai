@echo off
cd /d "%~dp0"
if not exist venv (
  echo Creating virtual environment...
  python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt
echo.
echo Starting Resume Screening AI at http://127.0.0.1:5000
start "" http://127.0.0.1:5000
python app.py
pause
