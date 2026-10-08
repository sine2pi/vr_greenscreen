@echo off
for %%i in (%*) do (
    ffmpeg -i "%%~fi" -vn -ac 2 -ar 44100 "%%~dpni.wav"
)
pause