@echo off
:loop
if "%~1"=="" goto end

echo Processing: "%~1"

ffprobe -v error -select_streams v:0 -count_frames ^
-show_entries stream=nb_read_frames -print_format csv ^
"%~1" 

shift
goto loop

:end
echo All files processed!
pause

