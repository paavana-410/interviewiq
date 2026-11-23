@echo off
cd /d C:\Users\Admin\Downloads\GEN\place\interviewiq
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
