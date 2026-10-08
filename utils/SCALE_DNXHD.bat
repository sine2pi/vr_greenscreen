@echo off
set /p "USER_FPS=Enter the number of frames per second (Leave blank to use source): "
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

if not "%USER_W%" == "" set "W=%USER_W%"
if not "%USER_H%" == "" set "H=%USER_H%"
if not "%USER_FPS%" == "" set "FPS=%USER_FPS%"
if "%FPS%" == "N/A" set "FPS=30"
if "%FPS%" == "0/0" set "FPS=30"


ffmpeg -y -i "%~1" ^
-map "[v]" -map "[a]" ^
-sws_flags bicubic+full_chroma_int+accurate_rnd+full_chroma_inp ^
-filter_complex "[0:v]fps=%FPS%,setpts=N/(%FPS%*TB),scale=w=%W%:h=%H%:flags=bicubic:out_range=pc:threads=0[v];[0:a]asetpts=N/SR/TB,aresample=async=1:min_comp=0.001:min_hard_comp=0.1:first_pts=0[a]" ^
-fps_mode cfr -r %FPS% ^
-strict experimental -c:v dnxhd -profile:v dnxhr_lb -pix_fmt yuv422p ^
-avoid_negative_ts make_zero -fflags +genpts -movflags +faststart ^
-color_primaries bt709 -color_trc bt709 -colorspace bt709 ^
-c:a aac -b:a 256k ^
-metadata:s:v:0 stereo_mode=left_right ^
-nostdin ^
"%~dpn1_DNX.mov"

shift
goto loop

:end
echo All files processed!
pause
