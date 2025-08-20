@echo off
echo Starting Chatbot Integration...
echo.

echo Starting Flask API Server...
start "Flask API Server" cmd /k "python api_server.py"

echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

echo Starting React Frontend...
start "React Frontend" cmd /k "npm run dev"

echo.
echo Chatbot is starting up!
echo - Flask API will be available at: http://localhost:5000
echo - React frontend will be available at: http://localhost:5173
echo.
echo Press any key to close this window...
pause > nul
