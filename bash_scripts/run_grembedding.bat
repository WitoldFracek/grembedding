@echo off

rem Wywołaj pierwszą komendę
echo Starting imports_validator...
python .\imports_validator\imports_validator.py

rem Sprawdź, czy pierwsza komenda zakończyła się sukcesem
if %errorlevel% equ 0 (
    rem Jeśli tak, wywołaj drugą komendę
    echo imports_validator ended without error
    echo Starting dvc...

    rem Env var do sterowania spacy
    set GRE_SPACY_MODE=cpu
    set GRE_SPACY_BATCH_SIZE=64

    rem Env var do sterowania douczaniem bertów
    set GRE_LARGE_MODEL_TRAIN_BATCH_SIZE=16
    set GRE_LARGE_MODEL_INFERENCE_BATCH_SIZE=64
    set GRE_FINE_TUNE_EPOCHS=5

    dvc exp run

    set PYTHONPATH=.
    python .\utils\mlflow\sync.py
) else (
    rem Jeśli pierwsza komenda nie powiodła się, wyświetl komunikat o błędzie
    echo Błąd podczas wykonania pierwszej komendy.
)


