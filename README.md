<pre>
<code>
<p style="text-align: center;">
 _____                    _              _     _ _             
|  __ \                  | |            | |   | (_)            
| |  \/_ __ ___ _ __ ___ | |__   ___  __| | __| |_ _ __   __ _ 
| | __| '__/ _ \ '_ ` _ \| '_ \ / _ \/ _` |/ _` | | '_ \ / _` |
| |_\ \ | |  __/ | | | | | |_) |  __/ (_| | (_| | | | | | (_| |
 \____/_|  \___|_| |_| |_|_.__/ \___|\__,_|\__,_|_|_| |_|\__, |
                                                          __/ |
                                                         |___/
</p>
</code>
</pre>

NIGDY NIE USUWAMY NIC Z PARAMSÓW!

1. Push **wszystkich** zmian na gita (wszystkich wszystkich nawet params.yaml)
2. Bez zrobienia żadnych zmian (ABSOLUTNIE ŻADNYCH!) `dvc exp run`
3. `git status`
4. `dvc add mlruns`
5. `dvc status --cloud` pownien pokazać zmiany w mlruns
6. `dvc push`
7. `git add .`
8. `git commit -m "message"`
9. `git push` \

Dla lubiących ryzyko:
* `git add . && git commit -m "push worktree" --allow-empty`
* `dvc exp run`
* `./after-dvc.sh` (dvc add, push + git add, push)

Na innym kompie
10. `git pull`
11. `dvc pull`