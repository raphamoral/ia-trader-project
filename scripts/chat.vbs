' Abre o chat com a LLM local (independente do painel e do laboratório).
' Abre também o Ollama, se não estiver aberto (modelos no R:).
Set sh = CreateObject("WScript.Shell")
sh.Environment("Process")("OLLAMA_MODELS") = "R:\Ollama\models"
sh.Run """R:\Ollama\ollama app.exe""", 0, False
sh.CurrentDirectory = "R:\WORK\AI-trader-project"
sh.Run """R:\WORK\AI-trader-project\.venv\Scripts\pythonw.exe"" -m trader.chat", 0, False
WScript.Sleep 2500
sh.Run "http://127.0.0.1:8766"
