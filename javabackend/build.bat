@echo off
echo Building Lipika Backend...
call mvn clean compile
if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build successful!
    echo.
    echo To run the application:
    echo   mvn spring-boot:run
    echo.
) else (
    echo.
    echo Build failed! Please check the errors above.
    pause
)

