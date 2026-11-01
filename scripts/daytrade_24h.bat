@echo off
rem Day trade / scalper ao vivo em papel, 24h. Reinicia sozinho se cair. NENHUMA ordem real.
rem Uso: daytrade_24h.bat <nome> <estilo: day|scalp> <features: none|white|all> <registro de bootstrap>
setlocal
set NAME=%1
set STYLE=%2
set FEATURES=%3
set BOOT=%4
cd /d R:\WORK\AI-trader-project
set TEMP=R:\WORK\AI-trader-project\data\tmp
set TMP=%TEMP%
if not exist data\live mkdir data\live
:loop
if exist data\live\%NAME%.STOP goto end
echo [%date% %time%] iniciando %NAME% (%STYLE%, %FEATURES%) >> data\live\%NAME%.log
.venv\Scripts\python.exe -m trader.daytrade live --style %STYLE% --features %FEATURES% --symbol SOL/USDT --extra-symbols ETH/USDT,XRP/USDT --db data\live\%NAME%.sqlite --bootstrap %BOOT% >> data\live\%NAME%.log 2>&1
if exist data\live\%NAME%.STOP goto end
echo [%date% %time%] processo terminou, reiniciando em 60s >> data\live\%NAME%.log
timeout /t 60 /nobreak > nul
goto loop
:end
echo [%date% %time%] parado pelo painel >> data\live\%NAME%.log
