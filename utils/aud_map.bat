@echo off
:loop
if "%~1" == "" goto end
echo Processing: "%~1"

set "AUDIO_FILE=%~dpn1.wav"

if not exist "%AUDIO_FILE%" (
    echo Error: Could not find "%AUDIO_FILE%"
    shift
    goto loop
)

ffmpeg -i "%~1" -i "%AUDIO_FILE%" ^
-filter_complex "[1:a]aresample=async=1:min_comp=0.001:min_hard_comp=0.1[a]" ^
-map 0:v -map "[a]" -c:v copy -c:a aac -b:a 256k ^
"%~dpn1_AMAP.mov"

shift
goto loop

:end
echo Process complete.
pause
