@echo off
:loop
if "%~1" == "" goto end
echo Processing: "%~1"

ffmpeg -y -hwaccel cuda -i "%~1" ^
-filter_complex "[0:v]fps=30,setpts=N/(30*TB),split=2[v1][v2];[v1]crop=iw/2:ih/2:iw/4:0[left];[v2]crop=iw/2:ih/2:iw/4:ih/2[right];[left][right]hstack=inputs=2,scale=2048:1024:out_range=full:flags=spline+accurate_rnd+full_chroma_int,format=p010le[v]" ^
 -map "[v]" -map 0:a? ^
-fps_mode cfr -r 30 ^
-c:v hevc_nvenc -profile:v main10 -pix_fmt p010le -rc cbr -b:v 150M -b_ref_mode disabled -tag:v hvc1 -g 30 -preset p6 ^
-c:a pcm_s24le -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
-avoid_negative_ts make_zero -fflags +genpts -movflags +faststart+write_colr+frag_keyframe+use_metadata_tags ^
-metadata:s:v:0 stereo_mode=left_right -nostdin ^
"%~dpn1_HEVC.mp4"

shift
goto loop
:end
echo All files processed!
pause