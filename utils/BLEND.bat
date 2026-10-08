@echo off
:loop
:: Check if there are no more files to process
if "%~1" == "" goto end

:: Ensure we have a pair of files
if "%~2"=="" (
    echo Error: You must select files in pairs.
    echo Missing second file for "%~1".
    pause
    exit /b
)

echo --------------------------------------------------
echo Processing: "%~1" and "%~2"

:: Clear variables from previous iterations so they don't carry over
set "A="
set "T="
set "S="
set "SKIP_A="
set "SKIP_B="
set "TIME_ARG="

set /p "A=Enter ratio of video A (Leave blank to use 0.5): "
set /p "T=Enter length in seconds (Leave blank to process full video): "
set /p "S=Enter where you want to start the clip in minutes (Leave blank to process full video, for one minute enter 01): "
set /p "SKIP_A=Enter frames to skip from START of Video A (Leave blank for 0): "
set /p "SKIP_B=Enter frames to skip from START of Video B (Leave blank for 0): "

if "%A%" == "" set "A=0.5"
if "%SKIP_A%" == "" set "SKIP_A=0"
if "%SKIP_B%" == "" set "SKIP_B=0"

:: Set TIME_ARG only if T was provided
if not "%T%" == "" set "TIME_ARG=-t %T%"

:: Formats the starting minutes safely for FFmpeg
if "%S%" == "" (
    set "START_TIME=00:00:00"
) else (
    set "START_TIME=00:%S%:00"
)

ffmpeg -y %TIME_ARG% -ss %START_TIME% -i "%~1" %TIME_ARG% -ss %START_TIME% -i "%~2" ^
-filter_complex "[0:v]select='gte(n,%SKIP_A%)'[v0];[1:v]select='gte(n,%SKIP_B%)'[v1];[v0][v1]blend=all_expr='A*%A%+B*(1-%A%)'[final]" ^
-map "[final]" -map 1:a? ^
-c:v hevc_qsv -profile:v main10 -pix_fmt p010le -b_ref_mode disabled -tag:v hvc1 -g 30 -rc cbr -b:v 120M ^
-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
-avoid_negative_ts make_zero -fflags +genpts -movflags +faststart+write_colr ^
-metadata:s:v:0 stereo_mode=left_right -c:a copy ^
-nostdin ^
"%~dpn1_B.mp4"

:: Shift twice to move past the two files we just processed
shift
shift
goto loop

:end
echo All files processed!
pause
