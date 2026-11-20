@echo off
rem Double-click launcher for the EMIT pipeline dashboard (binds 127.0.0.1 only).
cd /d "%~dp0.."
start "" http://127.0.0.1:8000
py -3.11 -m webapp %*
