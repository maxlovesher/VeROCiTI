@echo off
rem Puts the VeROCiTI board sketch on an Arduino Uno. Run it once per Uno, with the Uno plugged in by USB.
rem
rem     upload_sketch.bat            finds the Uno by itself
rem     upload_sketch.bat COM5       use this port (clone boards often aren't recognised by name)
rem     upload_sketch.bat check      only compile it, upload nothing
rem
rem It uses arduino-cli: either one on the PATH or the one that ships inside Arduino IDE 2.
rem If the dashboard is open, set LED Board > Connection > USB port to "Off" first so it lets go of the port.
setlocal EnableDelayedExpansion
set "SKETCH=%~dp0verociti_board"
set "FQBN=arduino:avr:uno"

set "CLI="
where arduino-cli >nul 2>nul && set "CLI=arduino-cli"
if not defined CLI for %%P in ("%ProgramFiles%\Arduino IDE" "%LOCALAPPDATA%\Programs\Arduino IDE") do (
  if exist "%%~P\resources\app\lib\backend\resources\arduino-cli.exe" set "CLI=%%~P\resources\app\lib\backend\resources\arduino-cli.exe"
)
if not defined CLI (
  echo.
  echo   X arduino-cli was not found. Install the Arduino IDE from https://www.arduino.cc/en/software
  echo     ^(it includes it^), then run this again. Or upload the sketch by hand: open
  echo     hardware\verociti_board\verociti_board.ino in the IDE, pick Board "Arduino Uno" and press Upload.
  exit /b 1
)

echo   ^> Checking the Arduino AVR board package
"%CLI%" core list 2>nul | findstr /i "arduino:avr" >nul
if errorlevel 1 (
  echo   ^> Installing it ^(first time only^)
  "%CLI%" core update-index
  "%CLI%" core install arduino:avr
  if errorlevel 1 ( echo   X Could not install arduino:avr ^(is this computer online?^). & exit /b 1 )
)

if /i "%~1"=="check" (
  echo   ^> Compiling only
  "%CLI%" compile --fqbn %FQBN% "%SKETCH%"
  exit /b !errorlevel!
)

set "PORT=%~1"
if not defined PORT (
  "%CLI%" board list > "%TEMP%\verociti_boards.txt" 2>nul
  for /f "tokens=1" %%A in ('findstr /i "arduino:avr:uno" "%TEMP%\verociti_boards.txt"') do if not defined PORT set "PORT=%%A"
)
if not defined PORT (
  echo.
  echo   X No Arduino Uno found. Plug it in with a USB DATA cable, or name the port:  upload_sketch.bat COM5
  echo     ^(Device Manager > Ports shows which COM number the Uno got^)
  exit /b 1
)

echo   ^> Uploading to %PORT% - the lamps will blink while it works
"%CLI%" compile --upload -p %PORT% --fqbn %FQBN% "%SKETCH%"
if errorlevel 1 (
  echo.
  echo   X The upload failed. Is the dashboard still holding the port? Set USB port to "Off" on the LED Board page,
  echo     or close it, then try again.
  exit /b 1
)
echo.
echo   Done. Open the dashboard, LED Board page: it should say "Arduino Uno on %PORT%".
exit /b 0
