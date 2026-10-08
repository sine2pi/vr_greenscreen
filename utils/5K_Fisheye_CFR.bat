@echo off
for %%i in (%*) do (
    ffmpeg -y -hwaccel cuda -i "%%~fi" ^
    -map 0:v:0 -map 0:a? ^
    -r 59.94 ^
    -fps_mode cfr ^
    -vf "v360=hequirect:fisheye:ih_fov=180:iv_fov=180:h_fov=180:v_fov=180:in_stereo=sbs:out_stereo=sbs,scale=4896:2448:flags=lanczos,fps=fps=59.94,decimate,setpts=N/59.94/TB" ^
    -c:v hevc_nvenc ^
    -rc cbr -b:v 100M -maxrate 100M -bufsize 100M ^
    -tag:v hvc1 -c:a copy -movflags +faststart ^
    -metadata:s:v:0 stereo_mode=left_right ^
    "%%~dpni_5K_Fisheye.mp4"
)
pause

