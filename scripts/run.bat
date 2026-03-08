@echo off
:: Heronix NeuroSim - Neural Network Simulator
:: Copyright (c) 2026 Heronix Education Systems LLC
:: Developed by Michael Katsaros

title Heronix NeuroSim v3.0

where java >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Java is not installed or not in your PATH.
    echo  Please install Java 17 or later from https://adoptium.net
    echo.
    pause
    exit /b 1
)

echo Starting Heronix NeuroSim...
java -jar "%~dp0..\dist\HeronixNeuroSim-3.0.0.jar" 2>nul

if %errorlevel% neq 0 (
    echo.
    echo  If the application failed to start, ensure you have Java 17+.
    echo  Check your version with: java -version
    echo.
    pause
)
