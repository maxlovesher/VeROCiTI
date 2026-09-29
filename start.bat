@echo off
title VeROCiTI Full-Stack Launcher
color 0b
echo ================================================================
echo           VEROCITI - VEHICLE LOCATION AND CITY TRAFFIC INTELLIGENCE
echo ================================================================
echo.
echo [1/2] Launching CityFlow Multi-Agent Python Server (Port 5000)...
start "CityFlow Backend Server" cmd /k "%~dp0start-backend.bat"

echo [2/2] Launching VeROCiTI React Dashboard (Port 5173)...
start "VeROCiTI React Vite" cmd /k "%~dp0start-frontend.bat"

echo.
echo ================================================================
echo   Both services are now running:
echo   - React Web Dashboard:  http://localhost:5173/
echo   - CityFlow API Backend: http://localhost:5000/
echo ================================================================
echo You can leave this window open or close it.
timeout /t 5 >nul
