@echo off

rem Wywołaj pierwszą komendę
echo Starting imports_validator...
python .\imports_validator\imports_validator.py

rem Sprawdź, czy pierwsza komenda zakończyła się sukcesem
if %errorlevel% equ 0 (
    rem Jeśli tak, wywołaj drugą komendę
    echo imports_validator ended without error
    echo Starting dvc...
    dvc exp run --ignore-errors
    rem TODO słabe to ignore errors bo nie widaomo kiedy expected a kiedy error - jednak chyba bym to jakoś przerobił żeby zawsze był output
) else (
    rem Jeśli pierwsza komenda nie powiodła się, wyświetl komunikat o błędzie
    echo Błąd podczas wykonania pierwszej komendy.
)

set PYTHONPATH=.
python .\utils\mlflow\sync.py
