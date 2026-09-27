@echo off
REM One-click demo: starts Redis, Neo4j and the dashboard.
REM Redis and Neo4j are portable installs in %TOOLS% (no Docker needed on this laptop).
if "%TOOLS%"=="" set TOOLS=E:\AIML\tools
cd /d %~dp0

echo Starting Redis...
start "Redis" /min "%TOOLS%\redis\redis-server.exe" --port 6379 --save "" --appendonly no

echo Starting Neo4j (takes ~15s)...
start "Neo4j" /min cmd /c "%TOOLS%\neo4j-community-5.26.12\bin\neo4j.bat console"

REM Open the browser once the app has had time to warm up.
start "" cmd /c "timeout /t 25 /nobreak >nul & start http://127.0.0.1:5000"

set NEO4J_WAIT_SECONDS=90
set PYTHONIOENCODING=utf-8
call venv\Scripts\activate
python run.py
