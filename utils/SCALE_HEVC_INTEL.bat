@echo off
set /p "USER_FPS=Enter the number of frames per second (Leave blank to use source): "
set /p "USER_BR=Enter Bit Rate (in Mbps, e.g. 10, (Leave blank to use source): "
set /p "USER_W=Enter W (e.g., 4096, Leave blank to use source): "
set /p "USER_H=Enter H (e.g., 2048, Leave blank to use source): "

:loop
if "%~1" == "" goto end
echo --------------------------------------------------
echo Processing: "%~1"

for /f "tokens=1,2,3 delims=," %%i in ('ffprobe -v error -select_streams v:0 -show_entries "stream=width,height,r_frame_rate" -of "csv=p=0" "%~1"') do (
    set "W=%%i"
    set "H=%%j"
    set "FPS=%%k"
)

for /f "delims=" %%l in ('ffprobe -v error -show_entries "format=bit_rate" -of "csv=p=0" "%~1"') do (
    set "BR=%%l"
)

if not "%USER_W%" == "" set "W=%USER_W%"
if not "%USER_H%" == "" set "H=%USER_H%"
if not "%USER_FPS%" == "" set "FPS=%USER_FPS%"

if "%FPS%" == "N/A" set "FPS=30"
if "%FPS%" == "0/0" set "FPS=30"

if not "%USER_BR%" == "" (
    set "FINAL_BR=%USER_BR%M"
) else (
    if "%BR%" == "N/A" (
        set "FINAL_BR=10M"
    ) else (
        set "FINAL_BR=%BR%"
    )
)


ffmpeg -y -i "%~1" ^
-sws_flags lanczos+full_chroma_int+accurate_rnd+full_chroma_inp ^
-filter_complex "[0:v]fps=%FPS%,setpts=N/(%FPS%*TB),scale=w=%W%:h=%H%:flags=lanczos:out_range=pc:threads=0[v]" ^
-map "[v]" -map a:0? ^
-fps_mode cfr -r %FPS% ^
-aspect 2:1 ^
-c:v hevc_qsv -profile:v main10 -pix_fmt p010le -tag:v hvc1 -g 30 -b:v %FINAL_BR% ^
-c:a copy ^
-vstats -copyts -start_at_zero -bitexact ^
-color_primaries bt709 -color_trc bt709 -colorspace bt709 ^
-movflags +faststart+write_colr+use_metadata_tags ^
-metadata:s:v:0 stereo_mode=left_right ^
-nostdin ^
"%~dpn1_10HEVC.mp4"

shift
goto loop

:end
echo All files processed!
pause