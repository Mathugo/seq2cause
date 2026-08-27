Output assets/demo.gif

Set Shell "zsh"
Set FontSize 18
Set Width 1000
Set Height 400
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 40ms

# Override the shell prompt before recording starts, so no real
# username/hostname (e.g. a corporate ID) ever appears in the GIF.
Hide
Type "export PS1='(.venv) seq2cause %% '"
Enter
Type "clear"
Enter
Show

Type "python scripts/demo_tqdm.py"
Enter
Sleep 9s
