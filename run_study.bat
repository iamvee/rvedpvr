@echo off
REM   run_study.bat          solve the queue
REM   run_study.bat 50       first 50 queued runs
REM   run_study.bat plan     build the queue only
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MAXCONC=16"
set "PERRUN_GB=2"
set "PERRUN_CPUS=1"
set "FORCECONC="
set "MAXRESTARTS=40"

set "ARG=%~1"
set "ARG2=%~2"
set "MODE="
set "SLICE="
if /I "%ARG%"=="plan" goto :argplan
if "%ARG%"=="" goto :argdone
set "T=%ARG:/=%"
if not "%T%"=="%ARG%" ( set "MODE=--shard %ARG%" & set "SLICE=1" & goto :argconc )
set "T=%ARG:-=%"
if not "%T%"=="%ARG%" ( set "MODE=--range %ARG%" & set "SLICE=1" & goto :argconc )
set "MODE=--limit %ARG%"
:argconc
set "FORCECONC=%ARG2%"
goto :argdone
:argplan
set "MODE=--plan-only"
if not "%ARG2%"=="" set "MODE=--plan-only --plan-scale %ARG2%"
:argdone

docker info >nul 2>&1
if errorlevel 1 (
  echo.
  echo *** Docker is not responding. Start Docker Desktop, wait until it says "Running",
  echo     then run this again. ***
  exit /b 1
)

set "VMGB="
for /f "usebackq delims=" %%M in (`docker info --format "{{.MemTotal}}" 2^>nul`) do set "MEMB=%%M"
if defined MEMB set "VMGB=!MEMB:~0,-9!"
if not defined VMGB set "VMGB=16"
if "!VMGB!"=="" set "VMGB=16"
set /a BUDGET=!VMGB! * 70 / 100
if !BUDGET! LSS 4 set /a BUDGET=4
set /a CONC=!BUDGET! / %PERRUN_GB%
if !CONC! GTR %MAXCONC% set /a CONC=%MAXCONC%
if !CONC! LSS 1 set "CONC=1"
if defined FORCECONC set "CONC=%FORCECONC%"
set /a MAXRAM=!CONC! * %PERRUN_GB%
echo Docker VM RAM: !VMGB! GB  ^|  running !CONC! solves in flight x %PERRUN_GB% GB = !MAXRAM! GB peak
if defined FORCECONC if !MAXRAM! GTR !VMGB! echo   ^(warning: !MAXRAM! GB requested vs !VMGB! GB in the VM - if runs die with exit 125, ask for fewer^)
set "EXTRA=--concurrency !CONC! --per-run-cpus %PERRUN_CPUS% --per-run-ram-gb %PERRUN_GB% --max-cpus !CONC! --max-ram-gb !MAXRAM! --run-retries 1"

call :ensure_image rvedpvr-fenics docker/solver.Dockerfile  || goto :err
call :ensure_image rvedpvr-runner docker/runner.Dockerfile  || goto :err

set /a TRY=0
:runloop
set /a TRY+=1
set "BEFORE=0"
for /f %%N in ('dir /s /b results\measurements.json 2^>nul ^| find /c "measurements.json"') do set "BEFORE=%%N"
echo.
echo [run] Launching %MODE%   (restart budget !TRY!/%MAXRESTARTS%, !CONC! concurrent) ...
docker rm -f rvedpvr-runner-c >nul 2>&1
docker run --rm --name rvedpvr-runner-c ^
  -e RUNNER_SELF=rvedpvr-runner-c ^
  -v /var/run/docker.sock:/var/run/docker.sock ^
  -v "%CD%:/work" -w /work ^
  rvedpvr-runner ^
  python run.py --image rvedpvr-fenics %EXTRA% %MODE%
set "RC=!errorlevel!"
if "!RC!"=="0" goto :done
if "!RC!"=="4" goto :fatal
if /I "%ARG%"=="plan" goto :err
set "AFTER=0"
for /f %%N in ('dir /s /b results\measurements.json 2^>nul ^| find /c "measurements.json"') do set "AFTER=%%N"
if !AFTER! GTR !BEFORE! set /a TRY=0
if !TRY! GEQ %MAXRESTARTS% goto :err
echo.
echo *** Runner exited (code !RC!) -- Docker/VM hiccup. Waiting 30s, then RESUMING
echo     from where it stopped (finished runs are skipped). Ctrl+C to stop. ***
ping -n 31 127.0.0.1 >nul 2>&1
goto :runloop

:done
echo.
echo Done.  Results: results\^<run_id^>\   summary: results\summary.csv
if defined SLICE echo This machine ran ONE SLICE. Collect every machine's results folder, then:
if defined SLICE echo     python merge_results.py --into results --from D:\boxB\results D:\boxC\results
goto :eof

:fatal
echo.
echo *** Stopped: the runner reported a CONFIGURATION error (see its message above).
echo     This is not a Docker problem and retrying cannot help. Either fix the
echo     argument, or copy the designs folder from the machine that ran
echo     "run_study.bat plan" so this box reads the same queue. ***
exit /b 4

:err
echo.
if not defined RC set "RC=%errorlevel%"
echo *** FAILED (exit !RC!). ***
echo     - Docker Desktop running? (whale icon "Running")
echo     - Details: designs\screen_report.json  and  results\*\run.log
exit /b 1

:ensure_image
docker image inspect %1 >nul 2>&1
if not errorlevel 1 (
  echo [img] %1 already present - skipping build.
  exit /b 0
)
echo [img] Building %1  (attempt 1/3) ...
docker build -t %1 -f %2 . && exit /b 0
echo [img] build failed - waiting 15s, then retrying ...
ping -n 16 127.0.0.1 >nul 2>&1
echo [img] Building %1  (attempt 2/3) ...
docker build -t %1 -f %2 . && exit /b 0
echo [img] build failed - waiting 15s, then retrying ...
ping -n 16 127.0.0.1 >nul 2>&1
echo [img] Building %1  (attempt 3/3) ...
docker build -t %1 -f %2 . && exit /b 0
echo.
echo *** Could not build %1 after 3 attempts.
echo     If the error mentions "contexts\meta\...\meta.json ... used by another process",
echo     that is a DOCKER DESKTOP file lock, not a problem with this study. Fix with any of:
echo         1^) close other terminals/windows running docker, then re-run this script
echo         2^) restart Docker Desktop ^(Troubleshoot ^> Restart^)
echo         3^) docker context use desktop-linux
echo     Then run this script again - it resumes, nothing is lost. ***
exit /b 1
