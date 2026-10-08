@echo off
setlocal DisableDelayedExpansion

:: 1. Force the script to execute inside the source folder to bypass System32 permissions
cd /d "%~dp1"

set "listfile=ffmpeg_list.txt"
if exist "%listfile%" del "%listfile%"

:: 2. Safely capture the name of the first file for output naming
for %%i in ("%~1") do (
    set "first_name=%%~ni"
    set "first_ext=%%~xi"
)

:: 3. Process files directly to avoid blowing up the line character limit
(for %%i in (%*) do (
    set "file=%%~fi"
    setlocal EnableDelayedExpansion
    :: Escape characters for FFmpeg's concat syntax
    set "file=!file:\=\\!"
    set "file=!file:'=\'!"
    echo file '!file!'
    endlocal
)) > "%listfile%"

:: 4. Execute concatenation via stream copy
ffmpeg -f concat -safe 0 -i "%listfile%" -c copy "_%first_name%%first_ext%"

:: 5. Cleanup
if exist "%listfile%" del "%listfile%"

echo.
echo Done! Merged video: "_%first_name%%first_ext%"
pause
