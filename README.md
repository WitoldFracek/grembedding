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

## Environment setup

1. Create a virtual environment with `python3 -m venv venv`. Supported versions: `3.10`, ...
2. Install dependencies using `pip install -r requirements.txt`.

## Requirements and versioning

### Use `venv` + `pip-tools` for dependency management.

1. Ensure you are using venv and have `pip-tools` installed.
2. Manage direct dependencies in `requirements.in`. You can pin the version to the newest one if you like.
Do not put transitive dependencies in `requirements.in` unless you want to pin them.
3. Use `pip-compile` to generate `requirements.txt`. This may take a bit to resolve all deps.
4. Use `pip-sync` to install the dependencies from `requirements.txt` into your virtual environment.

### Changing dependencies

1. Add / update / remove the dependency in `requirements.in`
2. Use `pip-compile && pip-sync` to sync venv with the new dependencies.
This is preferred to `pip install -r requirements.txt` because it will also uninstall unused dependencies.