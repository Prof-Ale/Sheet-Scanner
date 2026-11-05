@echo off
chcp 65001 >nul
title Sheet Scanner OMR - Conversor de Partituras
cd /d "%~dp0"

echo ========================================================
echo   Sheet Scanner OMR - Conversor de Partituras
echo   (Foto / PDF  --^>  MusicXML, MIDI e MuseScore 4)
echo ========================================================
echo.

echo [1/3] Liberando porta 7860 caso haja instancia anterior...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :7860 ^| findstr LISTENING') do taskkill /F /PID %%a >nul 2>&1

echo [2/3] Verificando bibliotecas Python...
python -m pip install -r requirements.txt --quiet

echo [3/3] Iniciando servidor local em http://localhost:7860 ...
start "" http://localhost:7860
python app.py
pause
