@echo off

rem Wywołaj pierwszą komendę
echo Starting imports_validator...
python .\imports_validator\imports_validator.py

rem Sprawdź, czy pierwsza komenda zakończyła się sukcesem
if %errorlevel% equ 0 (
    rem Jeśli tak, wywołaj drugą komendę
    echo imports_validator ended without error
    echo Starting dvc...

    set GRE_SPACY_MODE=gpu
    dvc exp run

    set PYTHONPATH=.
    python .\utils\mlflow\sync.py
) else (
    rem Jeśli pierwsza komenda nie powiodła się, wyświetl komunikat o błędzie
    echo Błąd podczas wykonania pierwszej komendy.
)


