@echo off
rem Evolução ao vivo em papel, 24h. Reinicia sozinho se o processo cair.
rem NENHUMA ordem real é enviada.
rem Uso: live_24h.bat <nome> <features: none|white|all>
rem Para parar: painel, ou criar data\live\<nome>.STOP
setlocal
set NAME=%1
set FEATURES=%2
if "%NAME%"=="" set NAME=continuo
if "%FEATURES%"=="" set FEATURES=none
cd /d R:\WORK\AI-trader-project
set TEMP=R:\WORK\AI-trader-project\data\tmp
set TMP=%TEMP%
if not exist data\live mkdir data\live
if not exist %TEMP% mkdir %TEMP%
:loop
if exist data\live\%NAME%.STOP goto end
echo [%date% %time%] iniciando %NAME% (%FEATURES%) >> data\live\%NAME%.log
.venv\Scripts\python.exe -m trader.live BTC/USDT 15m --window 3h --features %FEATURES% --db data\live\%NAME%.sqlite --extra-symbols ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT --memory data\evolution --bootstrap data\evolution\20260929-161409-s0.sqlite >> data\live\%NAME%.log 2>&1
if exist data\live\%NAME%.STOP goto end
echo [%date% %time%] processo terminou, reiniciando em 60s >> data\live\%NAME%.log
timeout /t 60 /nobreak > nul
goto loop
:end
echo [%date% %time%] parado pelo painel >> data\live\%NAME%.log
