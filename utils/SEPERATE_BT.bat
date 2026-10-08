@echo off
:loop
if "%~1" == "" goto end
echo Processing: "%~1"

ffmpeg -y -hwaccel cuda -i "%~1" ^
-vf "crop=w=iw/2:h=ih/2:x=iw/4:y=0;scale=w=1024:h=1024:flags=bicubic:out_range=pc:threads=0" ^
-strict experimental -c:v dnxhd -profile:v dnxhr_sq -pix_fmt yuv422p ^
-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
-c:a aac -b:a 256k ^
-movflags +faststart+write_colr+use_metadata_tags ^
-metadata:s:v:0 stereo_mode=left_right ^
"%~dpn1_TL.mov"

ffmpeg -y -hwaccel cuda -i "%~1" ^
-vf "crop=w=iw/2:h=ih/2:x=iw/4:y=ih/2;scale=w=1024:h=1024:flags=bicubic:out_range=pc:threads=0" ^
-strict experimental -c:v dnxhd -profile:v dnxhr_sq -pix_fmt yuv422p ^
-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
-c:a aac -b:a 256k ^
-movflags +faststart+write_colr+use_metadata_tags ^
-metadata:s:v:0 stereo_mode=left_right ^
"%~dpn1_BR.mov"
