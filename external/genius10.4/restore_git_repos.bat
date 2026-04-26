
REM Repository: projects/negmas/external/genius10.4
echo Restoring: projects/negmas/external/genius10.4
if exist "projects\negmas\external\genius10.4\.git" (
    echo   Directory already exists, skipping...
) else (
    REM Create parent directory if needed
    if not exist "projects\negmas\external" mkdir "projects\negmas\external"
    
    REM Clone the repository
    git clone "git@github.com:yasserfarouk/genius_source_code.git" "projects\negmas\external\genius10.4"
    if errorlevel 1 (
        echo   Failed to clone
    ) else (
        echo   Successfully cloned
        
        REM Checkout the original branch if not already on it
        cd "projects\negmas\external\genius10.4"
        git checkout "main" 2>nul
        if errorlevel 1 (
            echo   Could not checkout branch: main
        ) else (
            echo   Checked out branch: main
        )
        cd ..\..
    )
)
echo.


REM Repository: projects/negmas/external/genius10.4
echo Restoring: projects/negmas/external/genius10.4
if exist "projects\negmas\external\genius10.4\.git" (
    echo   Directory already exists, skipping...
) else (
    REM Create parent directory if needed
    if not exist "projects\negmas\external" mkdir "projects\negmas\external"
    
    REM Clone the repository
    git clone "git@github.com:yasserfarouk/genius_source_code.git" "projects\negmas\external\genius10.4"
    if errorlevel 1 (
        echo   Failed to clone
    ) else (
        echo   Successfully cloned
        
        REM Checkout the original branch if not already on it
        cd "projects\negmas\external\genius10.4"
        git checkout "main" 2>nul
        if errorlevel 1 (
            echo   Could not checkout branch: main
        ) else (
            echo   Checked out branch: main
        )
        cd ..\..
    )
)
echo.

