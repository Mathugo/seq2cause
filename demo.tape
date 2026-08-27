Output assets/demo.gif

Set Shell "zsh"
Set FontSize 15
Set Width 1000
Set Height 700
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 40ms

# Override the shell prompt before recording starts, so no real
# username/hostname (e.g. a corporate ID) ever appears in the GIF, and
# silence unrelated warnings (their tracebacks include the real absolute
# path to this checkout, which also leaks the same identifier).
Hide
Type "export PS1='(.venv) seq2cause %% ' PYTHONWARNINGS='ignore'"
Enter
Type "clear"
Enter
Show

Type "seq2cause --help"
Enter
Sleep 5s

Hide
Type "clear"
Enter
Show

Type "seq2cause --dataset examples/event_sequences.txt --model distilgpt2 --n-particles 2 --max-sequences 3 --seed 0"
Enter
Sleep 10s

