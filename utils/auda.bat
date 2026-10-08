@echo off
set /p "FRAMES=Enter the number of frames to delay (e.g., 28): "

:loop
if "%~1" == "" goto end
echo --------------------------------------------------
echo Processing: "%~1"

set "AUDIO_FILE=%~dpn1.m4a"

if not exist "%AUDIO_FILE%" (
    echo Error: Could not find "%AUDIO_FILE%"
    shift
    goto loop
)

:: Automatically pull the video frame rate using ffprobe
for /f "tokens=*" %%i in ('ffprobe -v error -select_streams v:0 -show_entries stream^=r_frame_rate -of default^=noprint_wrappers^=1:nocr^=1 "%~1"') do set "FPS_RAW=%%i"


for /f "tokens=1,2 delims=/" %%a in ("%FPS_RAW%") do (
    set "NUM=%%a"
    set "DEN=%%b"
)
if "%DEN%"=="" set "DEN=1"

:: Calculate millisecond delay using Windows Command Prompt math
set /a "MS=(%FRAMES% * 1000 * %DEN%) / %NUM%"

echo Detected Video FPS: %NUM%/%DEN%
echo Calculated Delay: %MS% ms

ffmpeg -y -hwaccel cuda -i "%~1" -i "%AUDIO_FILE%" ^
-filter_complex "[1:a]adelay=%MS%|%MS%[a]" ^
-map 0:v -map [a] -c:v copy -c:a aac -b:a 256k ^
"%~dpn1_aud.mp4"

shift
goto loop

:end
echo --------------------------------------------------
echo Process complete.
pause
