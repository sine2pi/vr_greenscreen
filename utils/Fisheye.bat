@echo off
for %%i in (%*) do (
ffmpeg -y -hwaccel cuda -i "%%~fi" ^
-filter_complex "[0:v]v360=hequirect:fisheye:ih_fov=180:iv_fov=180:h_fov=180:v_fov=180:in_stereo=sbs:out_stereo=sbs[v];[0:a]asetpts=N/SR/TB,aresample=async=1:min_comp=0.001:min_hard_comp=0.1:first_pts=0[a]" ^
-map "[v]" -map "[a]" ^
-sws_flags bicubic+full_chroma_int+accurate_rnd+full_chroma_inp ^
-fps_mode cfr ^
-c:v hevc_nvenc -profile:v main10 -pix_fmt p010le -b_ref_mode disabled -tag:v hvc1 -g 30 -rc cbr -b:v 60M -preset p6 ^
-c:a aac -b:a 256k ^
-color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range pc ^
-movflags +faststart+write_colr+use_metadata_tags ^
-metadata:s:v:0 stereo_mode=left_right ^
-nostdin ^
"%%~dpni_5K_Fisheye.mp4"
)
pause

