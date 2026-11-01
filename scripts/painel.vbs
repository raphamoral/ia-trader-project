' Abre o painel do AI Trader. Sobe o servidor sem janela (se já estiver no ar, a segunda cópia sai sozinha).
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "R:\WORK\AI-trader-project"
sh.Run """R:\WORK\AI-trader-project\.venv\Scripts\pythonw.exe"" -m trader.panel", 0, False
If WScript.Arguments.Count = 0 Then
  WScript.Sleep 1500
  sh.Run "http://127.0.0.1:8765"
End If
