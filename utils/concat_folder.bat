@echo off
setlocal EnableExtensions DisableDelayedExpansion

:: Check if a folder was dropped. If not, use the current directory.
if "%~1"=="" (
    set "target_dir=%cd%"
) else (
    if exist "%~1\" (
        set "target_dir=%~1"
    ) else (
        set "target_dir=%~dp1"
    )
)

set "listfile=%target_dir%\ffmpeg_list.txt"
if exist "%listfile%" del "%listfile%"

:: Set your video extension here (e.g., .mp4, .mkv, .ts)
set "ext=*.mp4"
set "first_name="

:: Loop through files in the target directory sorted alphabetically
for /f "delims=" %%a in ('dir "%target_dir%\%ext%" /b /o:n') do (
    set "file_name=%%a"
    set "full_path=%target_dir%\%%a"
    
    setlocal EnableDelayedExpansion
    :: Capture the first file's name for the output filename
    if not defined first_name (
        endlocal
        set "first_name=%%~na"
        setlocal EnableDelayedExpansion
    )
    
    :: Escape characters for FFmpeg's concat demuxer
    set "escaped=!full_path:\=\\!"
    set "escaped=!escaped:'=\'!"
    set "escaped=!escaped: =\ !"
    
    echo file '!escaped!'>>"%listfile%"
    endlocal
)

if not exist "%listfile%" (
    echo No %ext% files found in %target_dir%
    goto end
)

:: Run FFmpeg using the generated list
ffmpeg -f concat -safe 0 -i "%listfile%" -c copy "%target_dir%\_merged_%first_name%.mp4"

:: Clean up the text file
:: if exist "%listfile%" del "%listfile%"

echo.
echo Done! Merged video created in target folder.
:end
pause
