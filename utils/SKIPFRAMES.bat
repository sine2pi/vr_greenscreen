@echo off
set /p "FRAMES=Enter the number of frames to delay (e.g., 28): "

:loop
if "%~1" == "" goto end
echo --------------------------------------------------
echo Processing: "%~1"

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

ffmpeg -y -hwaccel cuda -ss %MS% -i "%~1" ^
 -map 0:v:0 -map 0:a? ^
 -fps_mode cfr ^
 -c:v hevc_nvenc -profile:v main10 -pix_fmt p010le -rc cbr -b:v 100M -b_ref_mode disabled -tag:v hvc1 -g 30 -tune hq ^
 -c:a aac -b:a 256k ^
 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
 -avoid_negative_ts make_zero -fflags +genpts ^
 -movflags +faststart+write_colr ^
 -metadata:s:v:0 stereo_mode=left_right ^
"%~dpn1_HEVC.mp4"

shift
goto loop

:end
echo All files processed!
pause