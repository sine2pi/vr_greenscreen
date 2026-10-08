@echo off
:loop
if "%~1" == "" goto end
echo Processing: "%~1"

ffmpeg -y -hwaccel cuda -i "%~1" ^
 -filter_complex "[0:v]crop=iw/2:ih/2:iw/4:0[left];[0:v]crop=iw/2:ih/2:iw/4:ih/2[right];[left][right]hstack=inputs=2[vo];[vo]scale=out_range=full:flags=spline+accurate_rnd+full_chroma_int,format=yuv422p10le[v]" ^
 -map "[v]" -map 0:a? -c:a copy ^
 -fps_mode cfr ^
 -c:v dnxhd -profile:v dnxhr_hqx -pix_fmt yuv422p10le ^
 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
 -movflags +faststart+write_colr+use_metadata_tags ^
 -metadata:s:v:0 stereo_mode=left_right ^
 "%~dpn1_dnxhd.mov"

shift
goto loop
:end
echo All files processed!
pause
