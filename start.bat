@echo off
rem VeROCiTI one-click start: double-click this (or the desktop button).
rem Everything happens in launch.ps1; this only opens it past PowerShell's script policy.
title VeROCiTI
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
if errorlevel 1 pause
